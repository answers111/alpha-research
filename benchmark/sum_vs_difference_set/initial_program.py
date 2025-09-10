import numpy as np

def main():
    N = 30
    # Sum-dominant example (maximize ratio): Conway MSTD set, take B=A
    A = [0, 2, 3, 4, 7, 11, 12, 14]
    B = A[:]
    A_ind = np.zeros(N, dtype=int); A_ind[A] = 1
    B_ind = np.zeros(N, dtype=int); B_ind[B] = 1
    print(f"N={N}, |A|={len(A)}, using MSTD A=B")
    return A_ind, B_ind

if __name__ == "__main__":
    A_indicators, B_indicators = main()

# Ensure compatibility with evaluators that expect globals
try:
    A_indicators; B_indicators  # type: ignore[name-defined]
except NameError:
    A_indicators, B_indicators = main()
