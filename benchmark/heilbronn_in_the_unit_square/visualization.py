# viz_min_triangle.py
import argparse
import importlib.util
import json
import os
import sys
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt


# ---------- 评测与辅助函数（与您 evaluator 一致/兼容） ----------

def _triangle_area(a, b, c) -> float:
    return abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])) * 0.5

def find_min_triangle(points: np.ndarray):
    """
    返回最小三角形：(i, j, k, min_area)
    若点数<3，返回 (-1, -1, -1, 0.0)
    """
    P = np.asarray(points, dtype=float)
    n = len(P)
    if n < 3:
        return -1, -1, -1, 0.0
    best = (-1, -1, -1, float("inf"))
    for i, j, k in combinations(range(n), 3):
        area = _triangle_area(P[i], P[j], P[k])
        if area < best[3]:
            best = (i, j, k, area)
            if area == 0.0:
                break
    return best

def evaluate_min_triangle_area(points: np.ndarray):
    """
    与您当前 evaluator 保持一致的指标：
      - min_area：最小三角形面积（越大越好）
      - scaled_min_area：n^(8/7 + 1/2000) * min_area
      - score：等于 min_area（越大越好）
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        return dict(valid=0.0, min_area=0.0, n=float(len(pts)),
                    scaled_min_area=0.0, score=0.0, argmin_triplet=(-1,-1,-1))
    i, j, k, min_area = find_min_triangle(pts)
    n = float(len(pts))
    exponent = (8.0/7.0) + (1.0/2000.0)
    scaled_min_area = (n ** exponent) * float(min_area)
    return dict(
        valid=1.0,
        min_area=float(min_area),
        n=n,
        scaled_min_area=float(scaled_min_area),
        score=float(min_area),
        argmin_triplet=(int(i), int(j), int(k))
    )


# ---------- 读取点数据（模块 / npy / csv） ----------

def load_from_module(module_path: str) -> np.ndarray:
    module_path = os.path.abspath(module_path)
    spec = importlib.util.spec_from_file_location("points_mod", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    pts = None
    if hasattr(mod, "points"):
        pts = mod.points
    elif hasattr(mod, "main"):
        res = mod.main()
        try:
            pts = np.asarray(res, dtype=float)
        except Exception:
            pass
    if pts is None and hasattr(mod, "points"):
        pts = mod.points
    if pts is None:
        raise RuntimeError("模块中既无 `points` 变量，也无法从 `main()` 获取点。")

    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"模块返回的点形状异常: {pts.shape}, 期望 (N,2)")
    return pts

def load_from_npy(npy_path: str) -> np.ndarray:
    pts = np.load(npy_path)
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"npy 形状异常: {pts.shape}, 期望 (N,2)")
    return pts

def load_from_csv(csv_path: str) -> np.ndarray:
    pts = np.loadtxt(csv_path, delimiter=",")
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"csv 形状异常: {pts.shape}, 期望 (N,2)")
    return pts


# ---------- 可视化 ----------

def plot_points_and_min_triangle(points: np.ndarray,
                                 show_indices: bool = False,
                                 title_prefix: str = ""):
    pts = np.asarray(points, dtype=float)
    (i, j, k, amin) = find_min_triangle(pts)
    eval_res = evaluate_min_triangle_area(pts)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")

    # 画所有点
    ax.scatter(pts[:, 0], pts[:, 1], s=40, zorder=2)

    # 可选：标注索引
    if show_indices:
        for idx, (x, y) in enumerate(pts):
            ax.text(x, y, str(idx), fontsize=9, ha="left", va="bottom")

    # 高亮最小三角形
    if i >= 0:
        tri = np.array([pts[i], pts[j], pts[k], pts[i]])
        ax.plot(tri[:, 0], tri[:, 1], linewidth=2.5, zorder=3)
        ax.scatter(pts[[i, j, k], 0], pts[[i, j, k], 1], s=70, zorder=4)

    # 网格与边框
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(True, linestyle="--", alpha=0.3)

    # 标题（包含指标）
    title = (
        f"{title_prefix}min_area={eval_res['min_area']:.8f} | "
        f"scaled_min_area={eval_res['scaled_min_area']:.6f} | "
        f"score={eval_res['score']:.8f} | "
        f"argmin={eval_res['argmin_triplet']}"
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

    # 同时在 stdout 打一份 JSON，方便脚本化调用时抓数值
    out = {
        "min_area": eval_res["min_area"],
        "scaled_min_area": eval_res["scaled_min_area"],
        "score": eval_res["score"],
        "argmin_triplet": eval_res["argmin_triplet"],
        "n": int(eval_res["n"]),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(
        description="可视化 [0,1]^2 中点集，并高亮最小三角形（支持模块 / .npy / .csv）。若未提供来源，将自动尝试读取同目录下的 points.npy"
    )
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument("--from-module", type=str, help="含 points 或 main() 的 Python 文件路径")
    src.add_argument("--from-npy", type=str, help="N×2 的 .npy 路径")
    src.add_argument("--from-csv", type=str, help="N×2 的 .csv 路径（逗号分隔）")
    parser.add_argument("--show-indices", action="store_true", help="是否标注点索引")
    parser.add_argument("--title", type=str, default="", help="标题前缀")
    args = parser.parse_args()

    if args.from_module:
        P = load_from_module(args.from_module)
    elif args.from_npy:
        P = load_from_npy(args.from_npy)
    elif args.from_csv:
        P = load_from_csv(args.from_csv)
    else:
        # 自动读取默认的 points.npy（位于本脚本同目录）
        default_path = os.path.join(os.path.dirname(__file__), "points.npy")
        if not os.path.exists(default_path):
            print(
                "未提供输入来源，且未在本目录找到 points.npy。请先运行 initial_program.py 生成 points.npy，或通过 --from-* 指定输入。",
                file=sys.stderr,
            )
            sys.exit(2)
        P = load_from_npy(default_path)

    # 可选：如果任务要求必须在 [0,1]^2，可以做个提示（不改变数值）
    if not (np.all(P >= 0.0) and np.all(P <= 1.0)):
        print("⚠️ 警告：存在越界点（不在 [0,1]^2），图中仍会显示。", file=sys.stderr)

    plot_points_and_min_triangle(P, show_indices=args.show_indices, title_prefix=args.title)

if __name__ == "__main__":
    main()
