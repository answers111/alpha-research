import numpy as np
import importlib.util
import sys
import os
import json
import traceback

def evaluate_uncertainty_area_simple(f: np.ndarray, tau: float = 1e-3):
    f = np.asarray(f)
    N = f.size
    if N == 0:
        return {"product": float("inf"), "Af": 0.0, "AF": 0.0,
                "k_time": 0, "k_freq": 0, "N": 0, "tau": float(tau)}
    F = np.fft.fft(f)
    mask_time = np.abs(f) > tau
    mask_freq = np.abs(F) > (tau * N)
    k_time = int(mask_time.sum())
    k_freq = int(mask_freq.sum())
    Af = k_time / N
    AF = k_freq / N
    product = Af * AF
    return {"product": float(product), "Af": float(Af), "AF": float(AF),
            "k_time": k_time, "k_freq": k_freq, "N": int(N), "tau": float(tau)}

def evaluate(program_path: str):
    """
    Load a program that provides a candidate vector x and return a single scalar:
    product if valid, else -1.0.
    """
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        sys.modules["program"] = program
        spec.loader.exec_module(program)

        x = None
        if hasattr(program, 'x'):
            x = program.x
        elif hasattr(program, 'main'):
            res = program.main()
            if isinstance(res, np.ndarray):
                x = res
            elif hasattr(program, 'x'):
                x = program.x

        if x is None:
            return -1.0

        result = evaluate_uncertainty_area_simple(x, tau=1e-3)
        if not np.isfinite(result["product"]) or result["N"] <= 0:
            return -1.0
        return float(result["product"])
    except Exception:
        return -1.0

if __name__ == "__main__":
    try:
        default_path = os.path.join(os.path.dirname(__file__), "initial_program.py")
    except Exception:
        default_path = "initial_program.py"
    target = sys.argv[1] if len(sys.argv) > 1 else default_path
    print(json.dumps(evaluate(target)))
