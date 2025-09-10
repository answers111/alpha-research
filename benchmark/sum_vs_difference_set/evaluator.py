import numpy as np
import importlib.util
import sys
import os
import json

EPS = 1e-30

def evaluate_ratio(A: np.ndarray, B: np.ndarray) -> float:
    """Return |A+B| / |A-B|; invalid returns -1.0."""
    Aidx = np.nonzero(np.asarray(A).astype(int))[0]
    Bidx = np.nonzero(np.asarray(B).astype(int))[0]
    if len(Aidx) == 0 or len(Bidx) == 0:
        return -1.0
    sumset, diffset = set(), set()
    for a in Aidx:
        for b in Bidx:
            sumset.add(a + b)
            diffset.add(a - b)
    if len(diffset) == 0:
        return -1.0
    return float(len(sumset) / (len(diffset) + EPS))

def evaluate(program_path: str):
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        sys.modules["program"] = program
        spec.loader.exec_module(program)

        A = None
        B = None
        # Prefer indicator arrays if present
        if hasattr(program, 'A_indicators') and hasattr(program, 'B_indicators'):
            A = program.A_indicators
            B = program.B_indicators
        elif hasattr(program, 'A') and hasattr(program, 'B'):
            A = program.A
            B = program.B
        elif hasattr(program, 'main'):
            res = program.main()
            if isinstance(res, tuple) and len(res) == 2:
                A, B = res
            elif hasattr(program, 'A_indicators') and hasattr(program, 'B_indicators'):
                A = program.A_indicators
                B = program.B_indicators
        if A is None or B is None:
            return -1.0
        ratio = evaluate_ratio(A, B)
        return float(ratio)
    except Exception:
        return -1.0

if __name__ == "__main__":
    try:
        default_path = os.path.join(os.path.dirname(__file__), "initial_program.py")
    except Exception:
        default_path = "initial_program.py"
    target = sys.argv[1] if len(sys.argv) > 1 else default_path
    print(json.dumps(evaluate(target)))
