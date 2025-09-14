import numpy as np
import subprocess
import sys
import traceback
import os
import json
from typing import Dict

def evaluate_C1_upper_std(step_heights: np.ndarray) -> Dict[str, float]:
    """
    Standard-normalized C1 evaluation function.
    - Project to feasible set: h >= 0 and ∫f = 1 (L1 normalization).
    - Objective: mu_inf = max_t (f*f)(t) (smaller is better).
    """
    h = np.asarray(step_heights, dtype=float)
    if h.size == 0 or np.any(h < 0):
        return {"valid": 0.0, "mu_inf": float("inf"), "ratio": float("inf")}
    K = int(len(h))
    dx = 1.0 / K
    integral = float(np.sum(h) * dx)
    if integral <= 0:
        return {"valid": 0.0, "mu_inf": float("inf"), "ratio": float("inf")}
    h = h / integral
    F = np.fft.fft(h, 2*K - 1)
    conv = np.fft.ifft(F * F).real
    conv = np.maximum(conv, 0.0)
    mu_inf = float(np.max(conv) * dx)
    return {"valid": 1.0, "mu_inf": mu_inf, "ratio": mu_inf, "integral": 1.0, "K": float(K)}

def evaluate(program_path: str):
    """
    Evaluate a program that solves the autoconvolution peak minimization problem.
    
    Returns: dict with key 'score' = 1 / mu_inf (larger is better), or {'error': -1.0}
    """
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        sys.modules["program"] = program
        spec.loader.exec_module(program)
        step_heights = None
        if hasattr(program, 'step_heights'):
            step_heights = program.step_heights
        elif hasattr(program, 'h'):
            step_heights = program.h
        elif hasattr(program, 'main'):
            result = program.main()
            if isinstance(result, np.ndarray):
                step_heights = result
            elif hasattr(program, 'step_heights'):
                step_heights = program.step_heights
            elif hasattr(program, 'h'):
                step_heights = program.h
        if step_heights is None:
            return {"error": -1.0}
        result = evaluate_C1_upper_std(step_heights)
        if result["valid"] == 1.0:
            mu = float(result.get("mu_inf", float("inf")))
            if mu > 0 and np.isfinite(mu):
                return {"score": 1.0 / mu}
            return {"error": -1.0}
        else:
            return {"error": -1.0}
    except Exception as e:
        return {"error": -1.0}
