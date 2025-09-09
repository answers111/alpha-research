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
    
    Args:
        program_path: Path to the Python program file to evaluate
        
    Returns:
        dict: Dictionary with metric names as keys and numeric scores as values
    """
    try:
        # Use importlib.util to dynamically load the program module
        import importlib.util
        
        # Load the module from the given path
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        sys.modules["program"] = program
        spec.loader.exec_module(program)
        
        # Look for step_heights or result in the loaded module
        step_heights = None
        if hasattr(program, 'step_heights'):
            step_heights = program.step_heights
        elif hasattr(program, 'h'):
            step_heights = program.h
        elif hasattr(program, 'main'):
            # If there's a main function, try calling it to get step_heights
            result = program.main()
            if isinstance(result, np.ndarray):
                step_heights = result
            elif hasattr(program, 'step_heights'):
                step_heights = program.step_heights
            elif hasattr(program, 'h'):
                step_heights = program.h
        
        if step_heights is None:
            return -1.0
        
        # Evaluate the step heights
        result = evaluate_C1_upper_std(step_heights)
        
        # Return single value: mu_inf if valid, -1 if invalid
        if result["valid"] == 1.0:
            return result["mu_inf"]
        else:
            return -1.0
        
    except Exception as e:
        return -1.0


if __name__ == "__main__":
    # CLI for debugging: evaluate initial_program.py by default, or a provided path
    try:
        default_path = os.path.join(os.path.dirname(__file__), "initial_program.py")
    except Exception:
        default_path = "initial_program.py"

    target = sys.argv[1] if len(sys.argv) > 1 else default_path
    print(json.dumps(evaluate(target), ensure_ascii=False, indent=2))
