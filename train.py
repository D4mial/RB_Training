#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Huấn luyện heuristic h(n) học được cho robot mê cung, so sánh với Manhattan trong GBFS / A*.

    python train.py                  # tự sinh dataset nếu chưa có
    python train.py --rebuild        # sinh lại dataset trước khi train

Đầu ra: models/heuristic_model.joblib, results/report.json, results/search_comparison.csv
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import build_dataset
from build_dataset import FEATURE_NAMES, cell_features
from robot_me_cung import SEED, a_star, bfs, greedy_best_first, manhattan

TRAIN, TEST, MAZES = "dataset/heuristic_train.csv", "dataset/heuristic_test.csv", "dataset/mazes.json"


def learned_heuristic(model, grid, goal, density):
    """Dự đoán h cho mọi ô trống của một mê cung, trả về hàm h(p, goal) tra bảng."""
    free = [(x, y) for y in range(len(grid)) for x in range(len(grid[0])) if grid[y][x] == 0]
    X = pd.DataFrame([cell_features(grid, p, goal, density) for p in free], columns=FEATURE_NAMES)
    table = dict(zip(free, np.maximum(model.predict(X), 0)))
    return lambda p, _goal: table.get(p, manhattan(p, goal))


def evaluate_search(model, data, test_ids):
    rows = []
    for mid in test_ids:
        m = data["mazes"][mid]
        g, s, t = m["grid"], tuple(m["start"]), tuple(m["goal"])
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
    ap.add_argument("--rebuild", action="store_true", help="sinh lại dataset trước khi train")
    a = ap.parse_args()

    if a.rebuild or not all(os.path.exists(f) for f in (TRAIN, TEST, MAZES)):
        print("Đang sinh dataset...")
        build_dataset.main([])

    train, test = pd.read_csv(TRAIN), pd.read_csv(TEST)
    model = HistGradientBoostingRegressor(max_iter=a.max_iter, learning_rate=a.lr,
                                          early_stopping=True, random_state=SEED)
    model.fit(train[FEATURE_NAMES], train["true_dist"])
    pred = model.predict(test[FEATURE_NAMES])
    yte = test["true_dist"]

    report = {
        "train_rows": len(train), "test_rows": len(test),
        "MAE_learned": float(mean_absolute_error(yte, pred)),
        "MAE_manhattan": float(mean_absolute_error(yte, test["manhattan"])),
        "R2_learned": float(r2_score(yte, pred)),
        "R2_manhattan": float(r2_score(yte, test["manhattan"])),
    }
    print("=== Sai số dự đoán khoảng cách tới đích (tập test) ===")
    for k, v in report.items():
        print(f"{k:>15}: {v:.3f}" if isinstance(v, float) else f"{k:>15}: {v}")

    with open(MAZES) as f:
        data = json.load(f)
    print("\n=== Tìm đường trên mê cung test (nút mở rộng TB, đường/tối ưu) ===")
    res = evaluate_search(model, data, data["split"]["test"])
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
