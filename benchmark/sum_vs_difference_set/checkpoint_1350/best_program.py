import numpy as np
from numpy.random import default_rng
import math
# Pre‐bind exponential for simulated‐annealing checks
exp = math.exp

import functools

rng = default_rng(42)
rng_random = rng.random
rng_choice = rng.choice

@functools.lru_cache(maxsize=None)
def _compute_ratio_bytes(A_bytes: bytes, B_bytes: bytes) -> float:
    A_int = np.frombuffer(A_bytes, dtype=np.int8).copy()
    B_int = np.frombuffer(B_bytes, dtype=np.int8).copy()
    if A_int.sum() == 0 or B_int.sum() == 0:
        return -1.0
    sum_counts = np.convolve(A_int, B_int)
    diff_counts = np.correlate(A_int, B_int, mode='full')
    num_sums = int((sum_counts != 0).sum())
    num_diffs = int((diff_counts != 0).sum())
    return -1.0 if num_diffs == 0 else num_sums / num_diffs

def compute_ratio(A_ind: np.ndarray, B_ind: np.ndarray) -> float:
    """Compute sum-to-difference ratio |A+B|/|A−B| via cached bytes-based LRU cache."""
    A_bytes = A_ind.astype(np.int8).tobytes()
    B_bytes = B_ind.astype(np.int8).tobytes()
    return _compute_ratio_bytes(A_bytes, B_bytes)

# Helper: perform one balanced swap/add/remove on a boolean indicator array
def propose_move(ind: np.ndarray) -> np.ndarray:
    """Randomly swap k bits between ones and zeros (k in {1,2,3}) with weighted probabilities."""
    ones = np.nonzero(ind)[0]
    zeros = np.nonzero(~ind)[0]
    if ones.size and zeros.size:
        # choose k ∈ {1,2,3} with probabilities 0.75, 0.2, 0.05
        k = rng_choice([1, 2, 3], p=[0.75, 0.2, 0.05])
        k = min(k, ones.size, zeros.size)
        removes = rng_choice(ones, size=k, replace=False)
        adds = rng_choice(zeros, size=k, replace=False)
        ind[removes] = False
        ind[adds] = True
    return ind

def get_local_best(current_A: np.ndarray, current_B: np.ndarray, batch_size: int = 10,
                   compute=compute_ratio, propose=propose_move, rand=rng_random):
    """Generate batch proposals and return the best local move and its ratio using local bindings."""
    local_best_ratio = -1.0
    local_best_A = current_A
    local_best_B = current_B
    for _ in range(batch_size):
        if rand() < 0.5:
            C_ind = propose(current_A.copy())
            D_ind = current_B.copy()
        else:
            C_ind = current_A.copy()
            D_ind = propose(current_B.copy())
        ratio = compute(C_ind, D_ind)
        if ratio > local_best_ratio:
            local_best_ratio = ratio
            local_best_A = C_ind
            local_best_B = D_ind
    return local_best_ratio, local_best_A, local_best_B

# Configuration constants
DEFAULT_N = 30
CONWAY_MSTD_INIT = [0, 2, 3, 4, 7, 11, 12, 14]

def main(N: int = DEFAULT_N) -> tuple[np.ndarray, np.ndarray]:
    """Perform hill‐climbing search starting from the Conway MSTD set of size N."""
    A_ind = np.zeros(N, dtype=bool)
    A_ind[CONWAY_MSTD_INIT] = True
    B_ind = A_ind.copy()

    # Evaluate initial ratio
    best_ratio = compute_ratio(A_ind, B_ind)
    best_A, best_B = A_ind.copy(), B_ind.copy()
    # Initialize simulated annealing
    current_A, current_B = best_A.copy(), best_B.copy()
    current_ratio = best_ratio
    T = 2.0             # raised initial temperature for broader exploration
    decay = 0.99990       # even slower cooling to maintain exploration

    # increase initial batch size for broader search
    batch_size = 50
    max_iter = 20000
    # Pre‐bind inner‐loop functions to locals for speed
    rng_rand = rng_random
    get_best = get_local_best
    for _ in range(max_iter):
        local_best_ratio, local_best_A, local_best_B = get_best(current_A, current_B, batch_size)
        # simulated annealing acceptance
        delta = local_best_ratio - current_ratio
        if delta > 0 or rng_rand() < exp(delta / T):
            current_ratio = local_best_ratio
            current_A, current_B = local_best_A, local_best_B
        # update global best
        if current_ratio > best_ratio:
            best_ratio, best_A, best_B = current_ratio, current_A, current_B
        # cool down
        T *= decay

    print(f"N={N}, best ratio={best_ratio:.4f}")
    return best_A, best_B

if __name__ == "__main__":
    A_ind, B_ind = main()
    print("A_ind:", A_ind)
    print("B_ind:", B_ind)
