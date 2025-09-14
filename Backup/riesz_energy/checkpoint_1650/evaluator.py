import sys
import os
import json
import importlib.util
import numpy as np
from typing import Dict

EPS = 1e-12

def evaluate_riesz_energy(points: np.ndarray, s: float = 1.0) -> Dict[str, float]:
    xs = np.clip(np.asarray(points, dtype=float).ravel(), 0.0, 1.0)
    xs.sort()
    n = len(xs)
    if n < 2:
        return {"valid": 0.0, "energy": float("inf"), "min_spacing": 0.0}
    energy = 0.0
    dmin = float("inf")
    for i in range(n):
        xi = xs[i]
        for j in range(i+1, n):
            d = abs(xi - xs[j])
            dmin = min(dmin, d)
            # 正确的夹断方式
            if d < EPS:
                d = EPS
            energy += 1.0 / (d ** s)
    return {"valid": 1.0, "energy": float(energy), "min_spacing": float(dmin)}


def evaluate(program_path: str = "benchmark/riesz_energy/checkpoint_1650/best_program.py"):
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    sys.modules["program"] = program
    spec.loader.exec_module(program)

    pts = None
    if hasattr(program, 'xs'):
        pts = program.xs
    elif hasattr(program, 'main'):
        res = program.main()
        if isinstance(res, np.ndarray):
            pts = res
        elif hasattr(program, 'xs'):
            pts = program.xs
    if pts is None:
        return {"error": -1.0}
    result = evaluate_riesz_energy(pts, s=1.0)
    if result.get("valid", 0.0) != 1.0:
        return {"error": -1.0}
    E = float(result["energy"])
    if E > 0 and np.isfinite(E):
        return {
            "energy": E,                 # 真·能量（越小越好）
            "score": 1.0 / E,            # 倒数分数（越大越好）
            "n": len(pts),
            "min_spacing": result["min_spacing"]
        }
    return {"error": -1.0}


print(evaluate())