import numpy as np


def main():
    N = 30
    # Conway MSTD set example; we take A=B for classical MSTD
    A = [0, 2, 3, 4, 7, 11, 12, 14]
    B = A[:]
    A_ind = np.zeros(N, dtype=int); A_ind[A] = 1
    B_ind = np.zeros(N, dtype=int); B_ind[B] = 1
    return A_ind, B_ind


# Ensure globals for evaluator
try:
    A_indicators; B_indicators  # type: ignore[name-defined]
except NameError:
    A_indicators, B_indicators = main()


