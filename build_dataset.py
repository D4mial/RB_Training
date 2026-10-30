#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh dataset huấn luyện heuristic cho robot mê cung (dùng robot_me_cung.py).

    python build_dataset.py [--per-density 80] [--perfect 60] [--cells 250]

Tạo ra thư mục dataset/:
  mazes.json            các mê cung + chia train/test (để đánh giá bằng GBFS/A*)
  heuristic_train.csv   mỗi dòng = 1 ô: đặc trưng + khoảng cách thật tới đích
  heuristic_test.csv
"""
import argparse
import json
import math
import os
import random
from collections import deque

import numpy as np
import pandas as pd

from robot_me_cung import DIRS, SEED, bfs, neighbors, perfect_maze, random_walls_maze

OUT = "dataset"
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


def maze_matrix(grid, goal, density, cells):
    """Trả về (X, nhãn y) cho các ô trong `cells` mà tới được G."""
    dist = true_distance_from_goal(grid, goal)
    pts = [p for p in cells if p in dist and p != goal]
    X = np.array([cell_features(grid, p, goal, density) for p in pts], dtype=np.float32)
    return X, np.array([dist[p] for p in pts], dtype=np.float32)


# --------------------------------------------------------------- Dataset
def make_mazes(rng, n_per_density, n_perfect):
    mazes = []
    for dens in (0.10, 0.20, 0.30):
        kind, count = f"K3_{int(dens * 100)}", 0
        while count < n_per_density:
            g, s, t = random_walls_maze(rng, 30, dens)
            if bfs(g, s, t)[0]:  # chỉ giữ mê cung giải được
                mazes.append({"kind": kind, "density": dens, "grid": g, "start": s, "goal": t})
                count += 1
    for _ in range(n_perfect):
        g, s, t = perfect_maze(rng, 31)
        mazes.append({"kind": "K4", "density": sum(map(sum, g)) / len(g) ** 2, "grid": g, "start": s, "goal": t})
    return mazes


def to_frame(mazes, ids, rng, cells_per_maze):
    frames = []
    for mid in ids:
        m = mazes[mid]
        g = m["grid"]
        free = [(x, y) for y in range(len(g)) for x in range(len(g[0])) if g[y][x] == 0]
        sample = rng.sample(free, min(cells_per_maze, len(free)))
        X, y = maze_matrix(g, tuple(m["goal"]), m["density"], sample)
        df = pd.DataFrame(X, columns=FEATURE_NAMES)
        df.insert(0, "maze_id", mid)
        df.insert(1, "kind", m["kind"])
        df["true_dist"] = y
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-density", type=int, default=80, help="số mê cung K3 mỗi mật độ")
    ap.add_argument("--perfect", type=int, default=60, help="số mê cung K4")
    ap.add_argument("--cells", type=int, default=250, help="số ô lấy mẫu mỗi mê cung")
    ap.add_argument("--test-ratio", type=float, default=0.25)
    a = ap.parse_args(argv)

    rng = random.Random(SEED)
    mazes = make_mazes(rng, a.per_density, a.perfect)
    ids = list(range(len(mazes)))
    rng.shuffle(ids)  # chia theo mê cung, không chia theo ô -> tránh rò rỉ dữ liệu
    n_test = int(len(ids) * a.test_ratio)
    split = {"test": sorted(ids[:n_test]), "train": sorted(ids[n_test:])}

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/mazes.json", "w") as f:
        json.dump({"mazes": mazes, "split": split}, f)
    for name in ("train", "test"):
        df = to_frame(mazes, split[name], rng, a.cells)
        df.to_csv(f"{OUT}/heuristic_{name}.csv", index=False)
        print(f"{name}: {len(split[name])} mê cung, {len(df)} dòng -> {OUT}/heuristic_{name}.csv")


if __name__ == "__main__":
    main()
