# viz_sphere_points.py
import argparse
import importlib.util
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

def _normalize_rows(P):
    nrm = np.linalg.norm(P, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    return P / nrm

def load_from_module(module_path: str):
    """
    动态加载模块：
    - 优先读取全局变量 `points`
    - 否则调用 `main()` 获取返回值
    """
    module_path = os.path.abspath(module_path)
    spec = importlib.util.spec_from_file_location("points_mod", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if hasattr(mod, "points"):
        pts = getattr(mod, "points")
    elif hasattr(mod, "main"):
        pts = mod.main()
    else:
        raise RuntimeError("模块中既无 `points` 变量，也无 `main()` 函数可获取点。")

    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"模块返回的点形状异常: {pts.shape}, 期望 (N, 3)")
    return _normalize_rows(pts)

def load_from_npy(npy_path: str):
    pts = np.load(npy_path)
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"npy 形状异常: {pts.shape}, 期望 (N, 3)")
    return _normalize_rows(pts)

def load_from_csv(csv_path: str):
    pts = np.loadtxt(csv_path, delimiter=",")
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"csv 形状异常: {pts.shape}, 期望 (N, 3)")
    return _normalize_rows(pts)

def min_pairwise_angle_deg(P):
    """
    返回：
    - 最小夹角（度）
    - 对应的最大余弦相似度（最近的一对点）
    """
    # 计算上三角的点积
    dot = P @ P.T
    n = len(P)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    vals = dot[mask]
    max_cos = np.max(vals) if vals.size else 1.0
    max_cos = np.clip(max_cos, -1.0, 1.0)
    ang_min = np.degrees(np.arccos(max_cos))
    return ang_min, max_cos

def plot_points_on_sphere(P, title="Spherical Point Set"):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1,1,1])

    # 画单位球面网格
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.15, linewidth=0, antialiased=True)

    # 画点
    ax.scatter(P[:,0], P[:,1], P[:,2], s=40, depthshade=True)

    # 坐标与视角
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="可视化 S^2 上的点集（从模块、.npy 或 .csv 读取）"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-module", type=str, help="包含 points 或 main() 的 .py 文件路径（例如 sphere_points.py）")
    src.add_argument("--from-npy", type=str, help="N×3 的 .npy 文件路径")
    src.add_argument("--from-csv", type=str, help="N×3 的 .csv 文件路径（逗号分隔）")
    parser.add_argument("--title", type=str, default="Spherical Point Set", help="图标题")
    args = parser.parse_args()

    if args.from_module:
        P = load_from_module(args.from_module)
    elif args.from_npy:
        P = load_from_npy(args.from_npy)
    else:
        P = load_from_csv(args.from_csv)

    ang_min_deg, max_cos = min_pairwise_angle_deg(P)
    print(f"N = {len(P)}")
    print(f"最小两点夹角 ≈ {ang_min_deg:.4f}°")
    print(f"最近对的余弦相似度（最大点积） ≈ {max_cos:.6f}")

    plot_points_on_sphere(P, title=args.title)

if __name__ == "__main__":
    main()
