#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Huấn luyện heuristic h(n) học được từ dataset ảnh mê cung, so sánh với Manhattan trong GBFS / A*.

    python train.py                  # tự sinh dataset ảnh nếu chưa có
    python train.py --rebuild        # sinh lại dataset ảnh trước khi train

Chia dữ liệu theo mê cung (ảnh): 75% đầu mỗi loại để train, 25% cuối để test.
K1 và K2 (cố định, 1 ảnh) chỉ dùng để kiểm tra.
Đầu ra: models/heuristic_model.joblib, results/report.json, results/search_comparison.csv
"""
import argparse
import glob
import json
import math
import os
import random
from collections import deque

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import build_dataset
from robot_me_cung import DIRS, SEED, a_star, bfs, greedy_best_first, manhattan, neighbors

IMAGES = "dataset/images"
FEATURE_NAMES = [
    "x", "y", "dx", "dy", "manhattan", "euclid",
    "wall_up", "wall_down", "wall_left", "wall_right",
    "free_neighbors", "wall_density_5x5", "wall_ratio_line", "maze_density", "maze_size",
]


# ------------------------------------------------------------ Đặc trưng
def true_distance_from_goal(grid, goal):
    """BFS ngược từ đích: khoảng cách thật từ mọi ô tới G (nhãn huấn luyện)."""
    dist, queue = {goal: 0}, deque([goal])
    while queue:
        u = queue.popleft()
        for v in neighbors(grid, u):
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def _is_wall(grid, x, y):
    h, w = len(grid), len(grid[0])
    return not (0 <= x < w and 0 <= y < h) or grid[y][x] == 1


def cell_features(grid, p, goal, density):
    x, y = p
    dx, dy = goal[0] - x, goal[1] - y
    walls = [_is_wall(grid, x + ddx, y + ddy) for ddx, ddy in DIRS]
    win = [_is_wall(grid, x + i, y + j) for i in range(-2, 3) for j in range(-2, 3)]
    steps = max(abs(dx), abs(dy), 1)  # lấy mẫu dọc đường thẳng p -> goal
    line = [_is_wall(grid, round(x + dx * t / steps), round(y + dy * t / steps)) for t in range(steps + 1)]
    return [x, y, dx, dy, abs(dx) + abs(dy), math.hypot(dx, dy),
            *map(int, walls), 4 - sum(walls), sum(win) / 25, sum(line) / len(line), density, len(grid)]


# ------------------------------------------------------------ Dữ liệu ảnh
def load_mazes(root=IMAGES):
    """Đọc toàn bộ ảnh trong dataset -> danh sách mê cung."""
    mazes = []
    for folder in sorted(glob.glob(os.path.join(root, "*"))):
        kind = os.path.basename(folder)
        for f in sorted(glob.glob(os.path.join(folder, "*.png"))):
            g, s, t = build_dataset.load_maze_image(f)
            mazes.append({"kind": kind, "file": f, "grid": g, "start": s, "goal": t,
                          "density": sum(map(sum, g)) / (len(g) * len(g[0]))})
    return mazes


def split_mazes(mazes, test_ratio):
    by_kind = {}
    for m in mazes:
        by_kind.setdefault(m["kind"], []).append(m)
    train, test = [], []
    for ms in by_kind.values():
        if len(ms) < 20:  # loại cố định (K1, K2): chỉ để kiểm tra
            test += ms
        else:
            k = int(len(ms) * (1 - test_ratio))
            train += ms[:k]
            test += ms[k:]
    return train, test


def to_frame(mazes, cells_per_maze, rng):
    frames = []
    for i, m in enumerate(mazes):
        g, t = m["grid"], m["goal"]
        dist = true_distance_from_goal(g, t)
        pts = [p for p in dist if p != t]
        pts = rng.sample(pts, min(cells_per_maze, len(pts)))
        X = [cell_features(g, p, t, m["density"]) for p in pts]
        df = pd.DataFrame(X, columns=FEATURE_NAMES)
        df.insert(0, "kind", m["kind"])
        df["true_dist"] = [dist[p] for p in pts]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------- Đánh giá tìm đường
def learned_heuristic(model, grid, goal, density):
    """Dự đoán h cho mọi ô trống của một mê cung, trả về hàm h(p, goal) tra bảng."""
    free = [(x, y) for y in range(len(grid)) for x in range(len(grid[0])) if grid[y][x] == 0]
    X = pd.DataFrame([cell_features(grid, p, goal, density) for p in free], columns=FEATURE_NAMES)
    table = dict(zip(free, np.maximum(model.predict(X), 0)))
    return lambda p, _goal: table.get(p, manhattan(p, goal))


def evaluate_search(model, mazes):
    rows = []
    for m in mazes:
        g, s, t = m["grid"], m["start"], m["goal"]
        opt = len(bfs(g, s, t)[0]) - 1
        h_ml = learned_heuristic(model, g, t, m["density"])
        for name, fn, h in (("GBFS-Manhattan", greedy_best_first, manhattan),
                            ("GBFS-Learned", greedy_best_first, h_ml),
                            ("A*-Manhattan", a_star, manhattan),
                            ("A*-Learned", a_star, h_ml)):
            path, expanded, _ = fn(g, s, t, h)
            rows.append({"kind": m["kind"], "algo": name, "nodes": len(expanded),
                         "ratio": (len(path) - 1) / opt if path and opt else np.nan})
    df = pd.DataFrame(rows)
    return df.groupby(["kind", "algo"]).agg(nodes=("nodes", "mean"), path_ratio=("ratio", "mean")).round(3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--cells", type=int, default=100, help="số ô lấy mẫu mỗi ảnh khi tạo dữ liệu huấn luyện")
    ap.add_argument("--test-ratio", type=float, default=0.25)
    ap.add_argument("--eval-per-kind", type=int, default=50, help="số mê cung test mỗi loại dùng để chạy GBFS/A*")
    ap.add_argument("--rebuild", action="store_true", help="sinh lại dataset ảnh trước khi train")
    a = ap.parse_args()

    if a.rebuild or not glob.glob(os.path.join(IMAGES, "*", "*.png")):
        print("Đang sinh dataset ảnh...")
        build_dataset.main([])

    mazes = load_mazes()
    train_m, test_m = split_mazes(mazes, a.test_ratio)
    print(f"Đã đọc {len(mazes)} ảnh: train {len(train_m)} mê cung, test {len(test_m)} mê cung")

    rng = random.Random(SEED)
    train, test = to_frame(train_m, a.cells, rng), to_frame(test_m, a.cells, rng)
    model = HistGradientBoostingRegressor(max_iter=a.max_iter, learning_rate=a.lr,
                                          early_stopping=True, random_state=SEED)
    model.fit(train[FEATURE_NAMES], train["true_dist"])
    pred, yte = model.predict(test[FEATURE_NAMES]), test["true_dist"]

    report = {
        "images": len(mazes), "train_mazes": len(train_m), "test_mazes": len(test_m),
        "train_rows": len(train), "test_rows": len(test),
        "MAE_learned": float(mean_absolute_error(yte, pred)),
        "MAE_manhattan": float(mean_absolute_error(yte, test["manhattan"])),
        "R2_learned": float(r2_score(yte, pred)),
        "R2_manhattan": float(r2_score(yte, test["manhattan"])),
    }
    print("\n=== Sai số dự đoán khoảng cách tới đích (tập test) ===")
    for k, v in report.items():
        print(f"{k:>15}: {v:.3f}" if isinstance(v, float) else f"{k:>15}: {v}")

    counts, eval_m = {}, []
    for m in test_m:  # lấy tối đa N mê cung test mỗi loại cho phần tìm đường
        if counts.get(m["kind"], 0) < a.eval_per_kind:
            counts[m["kind"]] = counts.get(m["kind"], 0) + 1
            eval_m.append(m)
    print(f"\n=== Tìm đường trên {len(eval_m)} mê cung test (nút mở rộng TB, đường/tối ưu) ===")
    res = evaluate_search(model, eval_m)
    print(res.to_string())

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    joblib.dump(model, "models/heuristic_model.joblib")
    with open("results/report.json", "w") as f:
        json.dump(report, f, indent=2)
    res.to_csv("results/search_comparison.csv")
    print("\nĐã lưu: models/heuristic_model.joblib, results/report.json, results/search_comparison.csv")


if __name__ == "__main__":
    main()
