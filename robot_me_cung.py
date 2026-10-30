#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    python robot_me_cung.py              # chạy thử K1, K2 và bảng thống kê K3, K4
    python robot_me_cung.py --demo k2    # xem bản đồ ASCII của một kịch bản (k1|k2|k3|k4)
    python robot_me_cung.py --full       # chạy đủ 200 mê cung K3 mỗi mật độ và 100 mê cung K4
    python robot_me_cung.py --gui        # mở giao diện mô phỏng robot (cần Tkinter)

Quy ước: lưới grid[y][x], 0 = ô trống, 1 = tường; tọa độ là (x, y).
"""
import argparse
import heapq
import math
import random
import sys
import time
from collections import deque

SEED = 2026
DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Lên, Xuống, Trái, Phải


# ---------------------------------------------------------------- Heuristic
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


HEURISTICS = {"manhattan": manhattan, "euclidean": euclidean}


# ------------------------------------------------------- Mô hình mê cung
def neighbors(grid, p):
    """Hàm chuyển trạng thái T(s, a): các ô kề hợp lệ."""
    h, w = len(grid), len(grid[0])
    for dx, dy in DIRS:
        x, y = p[0] + dx, p[1] + dy
        if 0 <= x < w and 0 <= y < h and grid[y][x] == 0:
            yield (x, y)


def trace(parent, start, goal):
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    return path[::-1]


# ------------------------------------------------------------- Bộ giải
# Mỗi bộ giải trả về (path, expanded, peak):
#   path     : danh sách ô từ S đến G, hoặc None nếu thất bại
#   expanded : danh sách ô đã được lấy ra xử lý (theo thứ tự) để tô màu / đếm nút
#   peak     : bộ nhớ đỉnh (số nút lưu đồng thời, OPEN + CLOSED)
def bfs(grid, start, goal, h=None):
    queue, parent, expanded = deque([start]), {start: None}, []
    while queue:
        u = queue.popleft()
        expanded.append(u)
        if u == goal:
            return trace(parent, start, goal), expanded, len(parent)
        for v in neighbors(grid, u):
            if v not in parent:
                parent[v] = u
                queue.append(v)
    return None, expanded, len(parent)


def greedy_best_first(grid, start, goal, h):
    """GBFS: f(n) = h(n). OPEN là heap, CLOSED là set."""
    open_q = [(h(start, goal), 0, start)]
    parent, closed, tie, expanded = {start: None}, set(), 0, []
    while open_q:
        _, _, u = heapq.heappop(open_q)
        if u in closed:
            continue
        expanded.append(u)
        if u == goal:
            return trace(parent, start, goal), expanded, len(parent)
        closed.add(u)
        for v in neighbors(grid, u):
            if v not in parent:  # chưa thuộc OPEN/CLOSED
                parent[v] = u
                tie += 1
                heapq.heappush(open_q, (h(v, goal), tie, v))
    return None, expanded, len(parent)


def a_star(grid, start, goal, h):
    """A*: f(n) = g(n) + h(n)."""
    g = {start: 0}
    parent, closed, tie, expanded = {start: None}, set(), 0, []
    open_q = [(h(start, goal), 0, start)]
    while open_q:
        _, _, u = heapq.heappop(open_q)
        if u in closed:
            continue
        expanded.append(u)
        if u == goal:
            return trace(parent, start, goal), expanded, len(g)
        closed.add(u)
        for v in neighbors(grid, u):
            ng = g[u] + 1
            if v not in g or ng < g[v]:
                g[v], parent[v] = ng, u
                tie += 1
                heapq.heappush(open_q, (ng + h(v, goal), tie, v))
    return None, expanded, len(g)


def hill_climbing(grid, start, goal, h):
    """Hill Climbing: chỉ sang láng giềng có h nhỏ hơn hẳn ô hiện tại."""
    current, path = start, [start]
    while current != goal:
        best, best_h = None, h(current, goal)
        for v in neighbors(grid, current):
            if h(v, goal) < best_h:
                best, best_h = v, h(v, goal)
        if best is None:
            return None, path, 1  # kẹt tại cực trị cục bộ
        current = best
        path.append(current)
    return path, path, 1


def hill_climbing_random_restart(grid, start, goal, h, rng, max_restarts=200):
    """HC + Random Restart: kẹt thì nhảy sang một ô trống ngẫu nhiên rồi leo tiếp.
    Trả về (thành công?, tổng số ô đã đi, số lần khởi động lại)."""
    free = [(x, y) for y in range(len(grid)) for x in range(len(grid[0])) if grid[y][x] == 0]
    steps, current = 0, start
    for r in range(max_restarts + 1):
        ok, visited, _ = hill_climbing(grid, current, goal, h)
        steps += len(visited)
        if ok:
            return True, steps, r
        current = rng.choice(free)
    return False, steps, max_restarts


# ------------------------------------------------------ Sinh mê cung
def open_maze(n=20):
    """K1: mê cung mở, không có vật cản."""
    return [[0] * n for _ in range(n)], (0, 0), (n - 1, n - 1)


def u_shaped_maze():
    """K2: bẫy chữ U (15 x 15), khe hở quay về phía xuất phát."""
    g = [[0] * 15 for _ in range(15)]
    for y in range(4, 11):
        g[y][8] = 1  # đáy chữ U
    for x in range(5, 9):
        g[4][x] = g[10][x] = 1  # hai cạnh chữ U
    return g, (1, 7), (13, 7)


def random_walls_maze(rng, n=30, density=0.2):
    """K3: rải tường ngẫu nhiên theo mật độ, S và G luôn là ô trống."""
    start, goal = (0, 0), (n - 1, n - 1)
    g = [[1 if rng.random() < density else 0 for _ in range(n)] for _ in range(n)]
    g[start[1]][start[0]] = g[goal[1]][goal[0]] = 0
    return g, start, goal


def perfect_maze(rng, n=31):
    """K4: mê cung hoàn hảo bằng quay lui (recursive backtracker, viết dạng vòng lặp).
    n phải là số lẻ; các 'ô phòng' nằm ở tọa độ lẻ, giữa hai phòng là một ô tường được đục."""
    g = [[1] * n for _ in range(n)]
    g[1][1] = 0
    stack = [(1, 1)]
    while stack:
        x, y = stack[-1]
        cand = [(x + dx * 2, y + dy * 2, dx, dy) for dx, dy in DIRS
                if 0 < x + dx * 2 < n - 1 and 0 < y + dy * 2 < n - 1 and g[y + dy * 2][x + dx * 2] == 1]
        if cand:
            nx, ny, dx, dy = rng.choice(cand)
            g[y + dy][x + dx] = 0  # đục tường giữa hai phòng
            g[ny][nx] = 0
            stack.append((nx, ny))
        else:
            stack.pop()  # hết đường: quay lui
    return g, (1, 1), (n - 2, n - 2)


# ------------------------------------------------------------ Hiển thị
def render(grid, start, goal, path=None, expanded=None, stuck=None):
    """Vẽ mê cung dạng chữ: # tường, S/G, * đường đi, . ô đã duyệt, X điểm kẹt."""
    cells = [["#" if c else " " for c in row] for row in grid]
    for (x, y) in (expanded or []):
        cells[y][x] = "."
    for (x, y) in (path or []):
        cells[y][x] = "*"
    if stuck:
        cells[stuck[1]][stuck[0]] = "X"
    cells[start[1]][start[0]] = "S"
    cells[goal[1]][goal[0]] = "G"
    return "\n".join("".join(r) for r in cells)


def run_one(name, grid, start, goal):
    """Chạy một thuật toán theo tên, trả về (path, expanded, peak, thời gian ms)."""
    algo, _, hname = name.partition(":")
    h = HEURISTICS.get(hname)
    fn = {"bfs": bfs, "gbfs": greedy_best_first, "astar": a_star, "hc": hill_climbing}[algo]
    t0 = time.perf_counter()
    path, expanded, peak = fn(grid, start, goal, h)
    return path, expanded, peak, (time.perf_counter() - t0) * 1000


ALGOS = [  # (nhãn hiển thị, tên nội bộ)
    ("BFS (tham chiếu)", "bfs"),
    ("GBFS - Manhattan", "gbfs:manhattan"),
    ("GBFS - Euclidean", "gbfs:euclidean"),
    ("HC - Manhattan", "hc:manhattan"),
    ("HC - Euclidean", "hc:euclidean"),
    ("A* - Manhattan", "astar:manhattan"),
]


# --------------------------------------------------------- Chế độ chạy
def demo(kind):
    rng = random.Random(SEED)
    scen = {
        "k1": ("K1: mê cung mở 20x20", lambda: open_maze(20)),
        "k2": ("K2: bẫy chữ U 15x15", u_shaped_maze),
        "k3": ("K3: tường ngẫu nhiên 30x30, mật độ 20%", None),
        "k4": ("K4: mê cung hoàn hảo 31x31", lambda: perfect_maze(rng, 31)),
    }[kind]
    if kind == "k3":  # lấy mê cung đầu tiên giải được
        while True:
            g, s, t = random_walls_maze(rng, 30, 0.2)
            if bfs(g, s, t)[0]:
                break
    else:
        g, s, t = scen[1]()
    print("=" * 70, "\n" + scen[0], "\n" + "=" * 70, sep="")
    for label, name in ALGOS:
        path, expanded, peak, ms = run_one(name, g, s, t)
        if path:
            msg = f"{len(path) - 1} bước"
        else:
            msg = "KẸT / không có đường"
        print(f"\n--- {label}: {msg}, mở rộng {len(expanded)} nút, bộ nhớ đỉnh {peak}, {ms:.3f} ms")
        if kind in ("k1", "k2") or label in ("GBFS - Manhattan", "HC - Manhattan"):
            stuck = expanded[-1] if not path and name.startswith("hc") else None
            print(render(g, s, t, path, expanded, stuck))


def table_k1_k2():
    print("=" * 70, "\nBẢNG K1 và K2 \n" + "=" * 70, sep="")
    print(f"{'Thuật toán':<20}{'K1 bước':>9}{'K1 nút':>9}{'K2 bước':>10}{'K2 nút':>9}")
    g1, s1, t1 = open_maze(20)
    g2, s2, t2 = u_shaped_maze()
    for label, name in ALGOS:
        p1, e1, _, _ = run_one(name, g1, s1, t1)
        p2, e2, _, _ = run_one(name, g2, s2, t2)
        b1 = str(len(p1) - 1) if p1 else "kẹt"
        b2 = str(len(p2) - 1) if p2 else "kẹt"
        print(f"{label:<20}{b1:>9}{len(e1):>9}{b2:>10}{len(e2):>9}")


def batch(title, mazes):
    """Chạy tất cả thuật toán trên một tập mê cung và in thống kê trung bình."""
    stats = {name: [0, 0.0, 0.0, 0.0, 0.0, 0] for _, name in ALGOS}  # ok, nút, bộ nhớ, ms, tỷ lệ tối ưu, số tỷ lệ
    rr_ok, rr_nodes, rng = 0, 0, random.Random(SEED)
    for g, s, t in mazes:
        opt = len(bfs(g, s, t)[0]) - 1
        for _, name in ALGOS:
            path, expanded, peak, ms = run_one(name, g, s, t)
            st = stats[name]
            st[1] += len(expanded); st[2] += peak; st[3] += ms
            if path:
                st[0] += 1
                st[4] += (len(path) - 1) / opt if opt else 1
                st[5] += 1
        ok, steps, _ = hill_climbing_random_restart(g, s, t, manhattan, rng)
        rr_ok += ok; rr_nodes += steps
    n = len(mazes)
    print("=" * 78, f"\n{title}  ({n} mê cung)\n" + "=" * 78, sep="")
    print(f"{'Thuật toán':<22}{'Thành công':>11}{'Nút TB':>9}{'Bộ nhớ TB':>11}{'ms TB':>9}{'Đường/tối ưu':>14}")
    for label, name in ALGOS:
        ok, nodes, mem, ms, ratio, cnt = stats[name]
        r = f"{ratio / cnt:.3f}" if cnt else "-"
        print(f"{label:<22}{100 * ok / n:>10.1f}%{nodes / n:>9.1f}{mem / n:>11.1f}{ms / n:>9.3f}{r:>14}")
    print(f"{'HC + Random Restart':<22}{100 * rr_ok / n:>10.1f}%{rr_nodes / n:>9.1f}{1.0:>11.1f}{'':>9}{'-':>14}")


def run_all(full):
    table_k1_k2()
    rng = random.Random(SEED)
    n3 = 200 if full else 40
    for dens in (0.10, 0.20, 0.30):
        mazes = []
        while len(mazes) < n3:  # chỉ giữ mê cung mà BFS tìm được đường
            g, s, t = random_walls_maze(rng, 30, dens)
            if bfs(g, s, t)[0]:
                mazes.append((g, s, t))
        batch(f"K3: tường ngẫu nhiên, mật độ {int(dens * 100)}%", mazes)
    n4 = 100 if full else 20
    batch("K4: mê cung hoàn hảo 31x31", [perfect_maze(rng, 31) for _ in range(n4)])
    if not full:
        print("\n(Đây là bản chạy nhanh.)")


# ----------------------------------------------------------- Giao diện
def run_gui():
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        sys.exit("Máy chưa có Tkinter. Hãy chạy chế độ chữ: python robot_me_cung.py --demo k2")

    rng = random.Random(SEED)
    root = tk.Tk()
    root.title("Robot tìm đường trong mê cung - GBFS và Hill Climbing")
    top = ttk.Frame(root, padding=6)
    top.pack(fill="x")
    scen_var = tk.StringVar(value="K2 - Bẫy chữ U")
    algo_var = tk.StringVar(value="GBFS")
    heur_var = tk.StringVar(value="manhattan")
    speed_var = tk.IntVar(value=30)
    for text, var, vals in (("Kịch bản", scen_var, ["K1 - Mê cung mở", "K2 - Bẫy chữ U", "K3 - Tường ngẫu nhiên 20%", "K4 - Mê cung hoàn hảo"]),
                            ("Thuật toán", algo_var, ["BFS", "GBFS", "Hill Climbing", "A*"]),
                            ("Heuristic", heur_var, ["manhattan", "euclidean"])):
        ttk.Label(top, text=text).pack(side="left", padx=(8, 2))
        ttk.Combobox(top, textvariable=var, values=vals, state="readonly", width=22 if text == "Kịch bản" else 13).pack(side="left")
    ttk.Label(top, text="Tốc độ").pack(side="left", padx=(8, 2))
    ttk.Scale(top, from_=1, to=100, variable=speed_var, length=90).pack(side="left")
    info = tk.StringVar(value="Chọn kịch bản và thuật toán rồi bấm Chạy.")
    ttk.Label(root, textvariable=info, padding=6).pack(fill="x")
    canvas = tk.Canvas(root, width=620, height=620, bg="white", highlightthickness=0)
    canvas.pack(padx=6, pady=6)

    state = {"job": None, "maze": None}

    def make_maze():
        key = scen_var.get()[:2]
        if key == "K1":
            return open_maze(20)
        if key == "K2":
            return u_shaped_maze()
        if key == "K3":
            while True:
                m = random_walls_maze(rng, 30, 0.2)
                if bfs(*m)[0]:
                    return m
        return perfect_maze(rng, 31)

    def draw_grid(g, s, t, cs):
        canvas.delete("all")
        for y, row in enumerate(g):
            for x, c in enumerate(row):
                if c:
                    canvas.create_rectangle(x * cs, y * cs, (x + 1) * cs, (y + 1) * cs, fill="#0B2A3C", outline="")
        for (x, y), col, lab in ((s, "#1F7A8C", "S"), (t, "#F28C28", "G")):
            canvas.create_oval(x * cs + 1, y * cs + 1, (x + 1) * cs - 1, (y + 1) * cs - 1, fill=col, outline="")
            if cs >= 14:
                canvas.create_text((x + .5) * cs, (y + .5) * cs, text=lab, fill="white")

    def new_maze():
        if state["job"]:
            root.after_cancel(state["job"])
        state["maze"] = make_maze()
        g, s, t = state["maze"]
        draw_grid(g, s, t, 620 // len(g))
        info.set("Đã tạo mê cung mới. Bấm Chạy.")

    def run():
        if state["job"]:
            root.after_cancel(state["job"])
        if state["maze"] is None:
            new_maze()
        g, s, t = state["maze"]
        cs = 620 // len(g)
        draw_grid(g, s, t, cs)
        name = {"BFS": "bfs", "GBFS": "gbfs:" + heur_var.get(), "Hill Climbing": "hc:" + heur_var.get(),
                "A*": "astar:" + heur_var.get()}[algo_var.get()]
        path, expanded, peak, ms = run_one(name, g, s, t)
        steps = list(expanded)

        def paint(cells, color):
            for (x, y) in cells:
                if (x, y) not in (s, t):
                    canvas.create_rectangle(x * cs + 1, y * cs + 1, (x + 1) * cs - 1, (y + 1) * cs - 1, fill=color, outline="")

        def step_explore(i=0):
            batch_n = max(1, speed_var.get() // 5)
            paint(steps[i:i + batch_n], "#C9D3DA")
            if i + batch_n < len(steps):
                state["job"] = root.after(max(1, 101 - speed_var.get()), step_explore, i + batch_n)
            else:
                finish()

        def finish():
            if path:
                paint(path, "#F28C28")
                info.set(f"Thành công: {len(path) - 1} bước, mở rộng {len(expanded)} nút, "
                         f"bộ nhớ đỉnh {peak}, {ms:.3f} ms.")
            else:
                x, y = expanded[-1]
                canvas.create_text((x + .5) * cs, (y + .5) * cs, text="X", fill="red", font=("Arial", max(10, cs)))
                info.set(f"KẸT sau {len(expanded) - 1} bước (cực trị cục bộ), mở rộng {len(expanded)} nút.")

        step_explore()

    ttk.Button(top, text="Mê cung mới", command=new_maze).pack(side="left", padx=(12, 2))
    ttk.Button(top, text="Chạy", command=run).pack(side="left", padx=2)
    new_maze()
    root.mainloop()


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Robot tìm đường trong mê cung (GBFS, Hill Climbing, BFS, A*)")
    ap.add_argument("--demo", choices=["k1", "k2", "k3", "k4"], help="in bản đồ ASCII của một kịch bản")
    ap.add_argument("--full", action="store_true", help="chạy đủ số mê cung  (200 mỗi mật độ K3, 100 K4)")
    ap.add_argument("--gui", action="store_true", help="mở giao diện mô phỏng (Tkinter)")
    a = ap.parse_args()
    if a.gui:
        run_gui()
    elif a.demo:
        demo(a.demo)
    else:
        run_all(a.full)
