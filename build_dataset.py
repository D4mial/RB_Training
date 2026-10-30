#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh dataset ảnh mê cung: mỗi mê cung là MỘT ảnh PNG (một mẫu dữ liệu).

    python build_dataset.py                 # mỗi loại 1000 mê cung
    python build_dataset.py --n 200         # số lượng nhỏ hơn để thử nhanh

Cấu trúc:
  dataset/images/K1/K1_0001.png            K1: mê cung mở 20x20 (cố định -> 1 ảnh)
  dataset/images/K2/K2_0001.png            K2: bẫy chữ U 15x15  (cố định -> 1 ảnh)
  dataset/images/K3_10/K3_10_0001.png ...  K3: tường ngẫu nhiên 30x30, mật độ 10% / 20% / 30%
  dataset/images/K4/K4_0001.png ...        K4: mê cung hoàn hảo 31x31

Mã hóa ảnh (mỗi ô lưới = CELL x CELL điểm ảnh):
  trắng = ô trống | đen = tường | xanh lá = xuất phát S | cam = đích G
"""
import argparse
import os
import random

import numpy as np
from PIL import Image

from robot_me_cung import SEED, bfs, open_maze, perfect_maze, random_walls_maze, u_shaped_maze

CELL = 10
FREE, WALL, START, GOAL = (255, 255, 255), (0, 0, 0), (0, 170, 0), (255, 140, 0)


# ------------------------------------------------------------ Ảnh <-> lưới
def save_maze_image(grid, start, goal, path):
    n, m = len(grid), len(grid[0])
    idx = np.array(grid, dtype=np.uint8)  # 0 = trống, 1 = tường
    idx[start[1]][start[0]] = 2
    idx[goal[1]][goal[0]] = 3
    big = idx.repeat(CELL, axis=0).repeat(CELL, axis=1)
    img = Image.fromarray(big, mode="P")
    img.putpalette([*FREE, *WALL, *START, *GOAL])
    img.save(path, optimize=True)


def load_maze_image(path):
    """Đọc ảnh PNG -> (grid, start, goal). Lấy màu tại tâm mỗi ô."""
    rgb = np.array(Image.open(path).convert("RGB"))
    h, w = rgb.shape[0] // CELL, rgb.shape[1] // CELL
    centers = rgb[CELL // 2::CELL, CELL // 2::CELL][:h, :w]
    grid = [[0] * w for _ in range(h)]
    start = goal = None
    for y in range(h):
        for x in range(w):
            c = tuple(int(v) for v in centers[y, x])
            if c == WALL:
                grid[y][x] = 1
            elif c == START:
                start = (x, y)
            elif c == GOAL:
                goal = (x, y)
    return grid, start, goal


# --------------------------------------------------------------- Sinh dữ liệu
def generators(rng):
    """(tên loại, hàm sinh, có ngẫu nhiên?) cho từng loại mê cung."""
    def k3(dens):
        def gen():
            while True:  # chỉ giữ mê cung BFS tìm được đường
                m = random_walls_maze(rng, 30, dens)
                if bfs(*m)[0]:
                    return m
        return gen
    return [
        ("K1", lambda: open_maze(20), False),
        ("K2", u_shaped_maze, False),
        ("K3_10", k3(0.10), True),
        ("K3_20", k3(0.20), True),
        ("K3_30", k3(0.30), True),
        ("K4", lambda: perfect_maze(rng, 31), True),
    ]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000, help="số mê cung mỗi loại ngẫu nhiên (K3 x3 mật độ, K4)")
    ap.add_argument("--out", default="dataset/images", help="thư mục lưu ảnh")
    a = ap.parse_args(argv)

    rng = random.Random(SEED)
    total = 0
    for kind, gen, is_random in generators(rng):
        folder = os.path.join(a.out, kind)
        os.makedirs(folder, exist_ok=True)
        count = a.n if is_random else 1  # K1, K2 cố định nên chỉ có 1 ảnh
        for i in range(1, count + 1):
            g, s, t = gen()
            save_maze_image(g, s, t, os.path.join(folder, f"{kind}_{i:04d}.png"))
        total += count
        print(f"{kind:<6}: {count:>5} ảnh -> {folder}/")
    print(f"Tổng: {total} ảnh")


if __name__ == "__main__":
    main()
