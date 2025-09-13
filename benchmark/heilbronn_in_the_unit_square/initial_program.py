import numpy as np
import os
import time
import argparse
from itertools import combinations

# =========================
# 基本几何与评估
# =========================

def triangle_area(a, b, c) -> float:
    # 三角形面积 = |(b-a) x (c-a)|/2
    return abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])) * 0.5

_triplets_cache = {}

def _precompute_triplets(n: int):
    from itertools import combinations as _comb
    combs = np.array(list(_comb(range(n), 3)), dtype=int)
    I, J, K = combs[:, 0], combs[:, 1], combs[:, 2]
    return I, J, K, combs

def find_min_triangle(P: np.ndarray):
    """向量化求最小三角形：返回 (i,j,k,min_area)。若 n<3 返回 (-1,-1,-1,0.0)。"""
    n = len(P)
    if n < 3:
        return -1, -1, -1, 0.0
    if n not in _triplets_cache:
        _triplets_cache[n] = _precompute_triplets(n)
    I, J, K, combs = _triplets_cache[n]
    A = P[I]
    B = P[J]
    C = P[K]
    area = np.abs((B[:,0]-A[:,0])*(C[:,1]-A[:,1]) - (B[:,1]-A[:,1])*(C[:,0]-A[:,0])) * 0.5
    if area.size == 0:
        return -1, -1, -1, 0.0
    idx = int(np.argmin(area))
    i, j, k = combs[idx]
    return int(i), int(j), int(k), float(area[idx])

def min_triangle_area(P: np.ndarray) -> float:
    return find_min_triangle(P)[3]

def scaled_min_area(P: np.ndarray) -> float:
    n = float(len(P))
    exp = (8.0/7.0) + (1.0/2000.0)
    return (n ** exp) * min_triangle_area(P)

# =========================
# 初始化：多起点
# =========================

def jittered_grid_points(n, seed=0):
    rng = np.random.default_rng(seed)
    m = int(round(np.sqrt(n))); m = max(m, 2)
    xs = (np.arange(m) + 0.5) / m
    ys = (np.arange(m) + 0.5) / m
    X, Y = np.meshgrid(xs, ys)
    P = np.c_[X.ravel(), Y.ravel()]
    jitter = 0.12 / m
    P += rng.uniform(-jitter, jitter, size=P.shape)
    P = np.clip(P[:n], 0.0, 1.0)
    return P

def hex_lattice_points(n, seed=0):
    rng = np.random.default_rng(seed)
    a = 1.0 / np.sqrt(n)
    pts = []
    y = a/2
    row = 0
    while y < 1.0:
        x0 = (a/2) if (row % 2 == 1) else a
        x = x0
        while x < 1.0:
            pts.append([x, y])
            x += a
        y += np.sqrt(3)/2 * a
        row += 1
    P = np.array(pts, dtype=float)
    if len(P) < n:
        extra = rng.uniform(0, 1, size=(n - len(P), 2))
        P = np.vstack([P, extra])
    P = P[:n]
    P += rng.uniform(-0.08*a, 0.08*a, size=P.shape)
    P = np.clip(P, 0.0, 1.0)
    return P

def bridson_poisson_disk(n, r=None, k=30, seed=0):
    """
    近似生成 >=n 的 Poisson-disk 点，再均匀抽样到 n 个。
    r: 目标最小间距 ~ c / sqrt(n)
    """
    rng = np.random.default_rng(seed)
    if r is None:
        r = 0.6 / np.sqrt(n)  # 稍保守的间距
    cell_size = r / np.sqrt(2)
    grid_w = int(np.ceil(1.0 / cell_size))
    grid_h = int(np.ceil(1.0 / cell_size))
    grid = -np.ones((grid_h, grid_w), dtype=int)

    def grid_coords(pt):
        return int(pt[1] / cell_size), int(pt[0] / cell_size)

    def in_neighborhood(pt):
        gy, gx = grid_coords(pt)
        for yy in range(max(gy-2,0), min(gy+3, grid_h)):
            for xx in range(max(gx-2,0), min(gx+3, grid_w)):
                j = grid[yy, xx]
                if j >= 0:
                    if np.linalg.norm(pts[j] - pt) < r:
                        return True
        return False

    pts = []
    active = []

    # 初始点
    p0 = rng.uniform(0, 1, size=2)
    pts.append(p0); active.append(0)
    gy, gx = grid_coords(p0); grid[gy, gx] = 0

    while active and len(pts) < max(n*2, n+10):
        idx = rng.choice(active)
        base = pts[idx]
        found = False
        for _ in range(k):
            rad = rng.uniform(r, 2*r)
            ang = rng.uniform(0, 2*np.pi)
            cand = base + rad * np.array([np.cos(ang), np.sin(ang)])
            if not (0 <= cand[0] <= 1 and 0 <= cand[1] <= 1):
                continue
            if not in_neighborhood(cand):
                pts.append(cand)
                gy, gx = grid_coords(cand); grid[gy, gx] = len(pts)-1
                active.append(len(pts)-1)
                found = True
                break
        if not found:
            active.remove(idx)

    pts = np.array(pts)
    if len(pts) >= n:
        idx = rng.choice(len(pts), size=n, replace=False)
        pts = pts[idx]
    else:
        extra = rng.uniform(0,1,size=(n-len(pts),2))
        pts = np.vstack([pts, extra])
    return pts

# =========================
# 定向局部搜索（增大最小三角形）
# =========================

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v

def bump_point(P, idx, step, rng):
    """对单个点做小扰动（含随机 + 轻微向内推），保持在 [0,1]^2。"""
    q = P.copy()
    jitter = rng.uniform(-step, step, size=2)
    # 轻微向内推，减少贴边导致的瘦三角形
    inward = 0.15 * step * (0.5 - P[idx])
    q[idx] = np.clip(P[idx] + jitter + inward, 0.0, 1.0)
    return q

def bump_min_triangle_directed(P, step, rng):
    """
    针对“当前最小三角形”的三个顶点，沿增大面积的几何方向优先移动：
    - 对顶点 a，相对边 (b,c) 的法向方向能增大面积。
    - 叠加小随机扰动，避免卡鞍点。
    """
    i, j, k, _ = find_min_triangle(P)
    if i < 0:
        return P
    a, b, c = P[i], P[j], P[k]

    def move_along_normal(P, idx, other1, other2):
        q = P.copy()
        base = other2 - other1
        # 2D 中与 base 垂直的法向（取两种方向试探）
        n1 = normalize(np.array([ base[1], -base[0] ]))
        n2 = -n1
        cand1 = np.clip(P[idx] + step*n1, 0.0, 1.0)
        cand2 = np.clip(P[idx] + step*n2, 0.0, 1.0)
        # 选择带来更大 min_area 的方向
        q1 = q.copy(); q1[idx] = cand1
        q2 = q.copy(); q2[idx] = cand2
        a1 = min_triangle_area(q1); a2 = min_triangle_area(q2)
        if a1 >= a2:
            return q1, a1
        else:
            return q2, a2

    # 依次尝试移动 i、j、k，并保留最好者
    bestP = P.copy(); bestA = min_triangle_area(P)
    for (idx, o1, o2) in [(i, b, c), (j, c, a), (k, a, b)]:
        q, area_dir = move_along_normal(bestP, idx, o1, o2)
        if area_dir > bestA + 1e-15:
            bestP, bestA = q, area_dir
        else:
            # 若定向无改进，退而求其次：随机小扰动
            q = bump_point(bestP, idx, 0.6*step, rng)
            a_rand = min_triangle_area(q)
            if a_rand > bestA + 1e-15:
                bestP, bestA = q, a_rand
    return bestP

def project_min_distance(P, dmin=1e-3, iters=1):
    """软约束：尽量避免过近点对（简单排斥迭代）。"""
    Q = P.copy()
    for _ in range(iters):
        for i in range(len(Q)):
            diffs = Q - Q[i]
            dists = np.linalg.norm(diffs, axis=1)
            mask = (dists < dmin) & (dists > 0)
            if np.any(mask):
                repel = -diffs[mask]
                move = 0.5 * np.sum(repel / np.maximum(dists[mask][:,None], 1e-12), axis=0)
                Q[i] = np.clip(Q[i] + 1e-3*move, 0.0, 1.0)
    return Q

def improve(P0, iters=6000, step0=0.05, seed=0, patience=800, time_limit=None):
    """
    退火式定向搜索：
    - 以“当前最小三角形”为线索，优先移动那三个点；
    - 步长逐步衰减并穿插随机扰动与最小距离投影；
    """
    rng = np.random.default_rng(seed)
    P = np.clip(P0.copy(), 0.0, 1.0)
    bestP = P.copy(); bestA = min_triangle_area(P)
    no_improve = 0
    t0 = time.time()

    step = step0
    for t in range(1, iters+1):
        if time_limit is not None and (time.time() - t0) > time_limit:
            break
        Q = bump_min_triangle_directed(P, step, rng)
        # 偶尔对非最小三角形顶点做随机扰动，避免局部陷阱
        if t % 30 == 0:
            idx = rng.integers(len(P))
            Q = bump_point(Q, idx, 0.5*step, rng)

        # 软性分离，避免过近
        if t % 50 == 0:
            Q = project_min_distance(Q, dmin=5e-3, iters=1)

        aQ = min_triangle_area(Q)
        if aQ > bestA + 1e-15:
            P = Q
            bestP, bestA = Q.copy(), aQ
            no_improve = 0
        else:
            # 以小概率接受较差解可加入，但这里保守：不接受
            no_improve += 1

        if no_improve >= patience:
            break

        # 步长衰减
        if t % 400 == 0:
            step *= 0.7
            step = max(step, 5e-4)

    return bestP, bestA

# =========================
# 主流程：多起点 + 精修
# =========================

def multi_start_optimize(n=16, seeds=(42, 43, 44), iters=6000, step0=0.05, time_limit=None):
    """
    多路起点（Hex / Grid / Poisson）并行，保留最好的，再额外精修一轮。
    """
    cands = []
    for s in seeds:
        cands.append(hex_lattice_points(n, seed=s))
        cands.append(jittered_grid_points(n, seed=1000+s))
        cands.append(bridson_poisson_disk(n, seed=2000+s))
    bestP = None; bestA = -1.0
    t0 = time.time()
    # 粗搜索：较少迭代，快速筛选
    coarse_iters = max(200, int(0.25 * iters))
    coarse_results = []
    for P0 in cands:
        remaining = None
        if time_limit is not None:
            elapsed = time.time() - t0
            remaining = max(0.0, time_limit - elapsed)
            if remaining <= 0:
                break
        P1, A1 = improve(P0, iters=coarse_iters, step0=step0, seed=12345, time_limit=remaining)
        coarse_results.append((A1, P1))

    if coarse_results:
        coarse_results.sort(key=lambda x: x[0], reverse=True)
        top_list = [P for (_, P) in coarse_results[:3]]
    else:
        top_list = cands[:1]

    # 精修：更小步长
    for idx, P0 in enumerate(top_list):
        remaining = None
        if time_limit is not None:
            elapsed = time.time() - t0
            remaining = max(0.0, time_limit - elapsed)
            if remaining <= 0:
                break
        P2, A2 = improve(P0, iters=max(400, int(0.6 * iters)), step0=0.02, seed=999+idx, time_limit=remaining)
        if A2 > bestA:
            bestP, bestA = P2, A2
    return bestP, bestA

# =========================
# 入口：生成 points 并保存
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--step0", type=float, default=0.04)
    parser.add_argument("--seeds", type=str, default="7,11,19")
    parser.add_argument("--time-limit", type=float, default=None)
    args = parser.parse_args()

    n = 16
    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip()) or (7, 11, 19)
    bestP, bestA = multi_start_optimize(n=n, seeds=seeds, iters=args.iters, step0=args.step0, time_limit=args.time_limit)
    smin = scaled_min_area(bestP)
    print(f"n={n}, points={len(bestP)}")
    print(f"min_area = {bestA:.10f}")
    print(f"scaled_min_area = {smin:.10f}")
    return bestP

if __name__ == "__main__":
    points = main()
    out_path = os.path.join(os.path.dirname(__file__), "points.npy")
    np.save(out_path, points)
    print(f"Saved points to {out_path}")

# 兼容外部 evaluator
try:
    points  # type: ignore[name-defined]
except NameError:
    points = main()
