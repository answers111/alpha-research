import sys
import re
import os
import json
import importlib.util
import numpy as np
from typing import Dict

def evaluate_littlewood_supnorm(coeffs, num_grid: int = 16384) -> Dict[str, float]:
    """
    FFT-sampled sup-norm upper bound on |P(e^{it})|.
    - coeffs: 1-D array-like of ±1
    - num_grid: sampling resolution (larger -> tighter upper bound)
    """
    if num_grid < 8:
        raise ValueError("num_grid too small")
    c = np.atleast_1d(np.asarray(coeffs, dtype=float))     # ensure 1-D
    if c.ndim != 1 or c.size == 0:
        raise ValueError("coeffs must be a non-empty 1-D array")

    pad = np.zeros(int(num_grid), dtype=np.complex128)
    pad[: c.size] = c.astype(np.complex128)

    values = np.fft.fft(pad)                               # samples of P on unit circle
    supnorm = float(np.max(np.abs(values)))
    return {"valid": 1.0, "supnorm": supnorm}

def _read_coeffs_from_stdin():
    """
    Accepts lines like:
      n = 512
      1 -1 1 1 -1 ...
    or just a line of ±1's. Robust to extra spaces/newlines.
    """
    text = sys.stdin.read().strip()
    if not text:
        return None
    # Take the last line that contains numbers
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # concatenate all non "n =" lines to support long sequences split across lines
    number_lines = [ln for ln in lines if not ln.lower().startswith("n =")]
    if not number_lines:
        return None
    joined = " ".join(number_lines)
    nums = re.findall(r"[-+]?\d+", joined)
    return np.asarray(list(map(int, nums)), dtype=int)

def evaluate(program_path: str = "littlewood_polynomials/initial_program.py"):
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        sys.modules["program"] = program
        spec.loader.exec_module(program)

        coeffs_obj = None
        if hasattr(program, 'coeffs'):
            coeffs_obj = program.coeffs
        elif hasattr(program, 'main'):
            res = program.main()
            if isinstance(res, np.ndarray):
                coeffs_obj = res
            elif hasattr(program, 'coeffs'):
                coeffs_obj = program.coeffs
        if coeffs_obj is None:
            # fallback: try stdin for robustness
            coeffs_obj = _read_coeffs_from_stdin()
            if coeffs_obj is None:
                return {"error": -1.0}
        result = evaluate_littlewood_supnorm(coeffs_obj, num_grid=16384)
        if result.get("valid", 0.0) != 1.0:
            return {"error": -1.0}
        supnorm = float(result["supnorm"])
        if supnorm > 0 and np.isfinite(supnorm):
            return {"score": 1.0 / supnorm}
        return {"error": -1.0}
    except Exception:
        return {"error": -1.0}

print(evaluate())