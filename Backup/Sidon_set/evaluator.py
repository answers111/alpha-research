import numpy as np
import importlib.util
import sys
import os
import json
import traceback
from typing import Dict

def evaluate_sidon_size(indicators: np.ndarray) -> Dict[str, float]:
    x = np.asarray(indicators).astype(int)
    idxs = np.nonzero(x)[0] + 1
    seen, violations = set(), 0
    for i in range(len(idxs)):
        a = idxs[i]
        for j in range(i, len(idxs)):
            b = idxs[j]
            s = a + b
            if s in seen:
                violations += 1
            else:
                seen.add(s)
    size = len(idxs)
    score = size if violations == 0 else -float(violations)
    return {
        "valid": 1.0 if violations == 0 else 0.0,
        "size": float(size),
        "score": score,
        "violations": float(violations),
    }

def evaluate(program_path: str):
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        sys.modules["program"] = program
        spec.loader.exec_module(program)

        indicators = None
        if hasattr(program, 'indicators'):
            indicators = program.indicators
        elif hasattr(program, 'main'):
            res = program.main()
            if isinstance(res, np.ndarray):
                indicators = res
            elif hasattr(program, 'indicators'):
                indicators = program.indicators

        if indicators is None:
            return -1.0

        result = evaluate_sidon_size(indicators)
        if result["valid"] == 1.0:
            return float(result["size"])  # maximize size
        else:
            return -1.0
    except Exception:
        return -1.0

if __name__ == "__main__":
    try:
        default_path = os.path.join(os.path.dirname(__file__), "initial_program.py")
    except Exception:
        default_path = "initial_program.py"
    target = sys.argv[1] if len(sys.argv) > 1 else default_path
    print(json.dumps(evaluate(target)))
