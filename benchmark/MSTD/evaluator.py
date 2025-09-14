import numpy as np
import importlib.util
import sys
import os
import json


def _to_index_set(indicators):
    arr = np.asarray(indicators).astype(int)
    return np.nonzero(arr)[0]


def mstd_ratio(A_idx, B_idx=None):
    """
    Classical MSTD ratio with enforced A=B: R = |A+A| / |A−A|.
    Any provided B is ignored to keep consistency with the baseline setting.
    """
    # Enforce classic setting: ignore B and set B_idx = A_idx
    B_idx = A_idx
    if len(A_idx) == 0 or len(B_idx) == 0:
        return -1.0
    sumset, diffset = set(), set()
    for a in A_idx:
        for b in B_idx:
            sumset.add(int(a + b))
            diffset.add(int(a - b))
    if len(diffset) == 0:
        return -1.0
    return float(len(sumset)) / float(len(diffset))


def evaluate(program_path: str):
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        sys.modules["program"] = program
        spec.loader.exec_module(program)

        # Accept either A_indicators (and optional B_indicators) or a main() returning them
        A = None
        B = None
        if hasattr(program, 'A_indicators'):
            A = program.A_indicators
        if hasattr(program, 'B_indicators'):
            B = program.B_indicators
        if A is None:
            if hasattr(program, 'main'):
                res = program.main()
                if isinstance(res, tuple) and len(res) in (1, 2):
                    if len(res) == 1:
                        A = res[0]
                        B = None
                    else:
                        A, B = res
        if A is None:
            return {"error": -1.0}

        A_idx = _to_index_set(A)
        # Enforce classic setting regardless of provided B
        R = mstd_ratio(A_idx, None)
        if R <= 0:
            return {"error": -1.0}
        # Higher is better: score = R
        return {"score": float(R), "ratio": float(R)}
    except Exception:
        return {"error": -1.0}


if __name__ == "__main__":
    try:
        default_path = os.path.join(os.path.dirname(__file__), "initial_program.py")
    except Exception:
        default_path = "initial_program.py"
    target = sys.argv[1] if len(sys.argv) > 1 else default_path
    print(json.dumps(evaluate(target), ensure_ascii=False))


