import numpy as np


def main():
    N = 30
    # Contiguous A=B gives |A+A| = |A-A|, so R = 1 (perfect balance)
    k = 8
    A = list(range(k))
    B = A[:]
    A_ind = np.zeros(N, dtype=int); A_ind[A] = 1
    B_ind = np.zeros(N, dtype=int); B_ind[B] = 1
    return A_ind, B_ind


# Ensure globals for evaluator
try:
    A_indicators; B_indicators  # type: ignore[name-defined]
except NameError:
    A_indicators, B_indicators = main()


