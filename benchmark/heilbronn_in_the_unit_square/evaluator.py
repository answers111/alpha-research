import numpy as np
import importlib.util
import sys
import os
import json
from itertools import combinations
from typing import Dict


def _triangle_area(a, b, c) -> float:
    return abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])) * 0.5


def evaluate_min_triangle_area(points: np.ndarray) -> Dict[str, float]:
    """
    Compute the minimum triangle area for 2D points.
    Metrics:
      - min_area: raw smallest triangle area (larger is better)
      - scaled_min_area: n^(8/7 + 1/2000) * min_area (informational)
      - score: equals min_area (larger is better)
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return {"valid": 0.0, "min_area": 0.0, "n": 0.0, "scaled_min_area": 0.0, "score": 0.0}

    n = len(pts)
    if n < 3:
        return {"valid": 0.0, "min_area": 0.0, "n": float(n), "scaled_min_area": 0.0, "score": 0.0}

    min_area = float("inf")
    for i, j, k in combinations(range(n), 3):
        area = _triangle_area(pts[i], pts[j], pts[k])
        if area < min_area:
            min_area = area
            if min_area <= 1e-18:  # 早停
                break

    exponent = (8.0/7.0) + (1.0/2000.0)
    scaled_min_area = (n ** exponent) * float(min_area)
    score = float(min_area) if np.isfinite(min_area) else 0.0

    return {
        "valid": 1.0,
        "min_area": float(min_area),
        "n": float(n),
        "scaled_min_area": float(scaled_min_area),
        "score": score,
    }



def evaluate(program_path: str):
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        sys.modules["program"] = program
        spec.loader.exec_module(program)

        points = None
        if hasattr(program, 'points'):
            points = program.points
        elif hasattr(program, 'main'):
            res = program.main()
            if isinstance(res, np.ndarray):
                points = res
            elif hasattr(program, 'points'):
                points = program.points
        if points is None:
            return {"error": -1.0}
        result = evaluate_min_triangle_area(points)
        # Return both the diagnostic dict and rely on 'score' as the main metric (larger is better)
        return {
            "score": result["score"],
            "scaled_min_area": result["scaled_min_area"],
            "min_area": result["min_area"],
            "n": result["n"],
        }
    except Exception:
        return {"error": -1.0}


if __name__ == "__main__":
    try:
        default_path = os.path.join(os.path.dirname(__file__), "initial_program.py")
    except Exception:
        default_path = "initial_program.py"
    target = sys.argv[1] if len(sys.argv) > 1 else default_path
    print(json.dumps(evaluate(target), ensure_ascii=False, indent=2))
