import numpy as np
import random
from typing import List, Tuple

def build_sidon_greedy(K: int, order: List[int]) -> List[int]:
    """
    Single greedy construction:
    - Maintain a set of 'used sums'
    - Traverse elements of {1,...,K} in the given order
    - Add an element t to A only if it does not create a duplicate sum
    - Greedy intuition: avoid collisions early, usually produces |A| ~ sqrt(K)
    """
    A = []
    sums_seen = set()
    for t in order:
        ok = True
        # Check if adding t will conflict with any existing element in A
        for a in A:
            if a + t in sums_seen:
                ok = False
                break
        if ok:
            # Safe to add t → update all sums
            for a in A:
                sums_seen.add(a + t)
            sums_seen.add(t + t)  # self-pair sum
            A.append(t)
    return A

def improve_by_order_shake(K: int, base_seed: int, restarts: int = 32) -> Tuple[List[int], List[int]]:
    """
    Multi-restart greedy with order shaking:
    - Randomize the order of numbers 1..K and run greedy multiple times
    - Also try a small perturbation of the order to escape bad local choices
    - Keep the largest set found
    Why? The Sidon greedy result depends heavily on order, so restarts help.
    """
    rng = random.Random(base_seed)
    best_A, best_order = [], list(range(1, K + 1))

    for r in range(restarts):
        order = list(range(1, K + 1))
        rng.shuffle(order)
        A = build_sidon_greedy(K, order)
        if len(A) > len(best_A):
            best_A, best_order = A, order

        # Small perturbation: move the first m elements to the end and rebuild
        m = rng.randint(2, min(25, K // 4))
        shaken = order[m:] + order[:m]
        A2 = build_sidon_greedy(K, shaken)
        if len(A2) > len(best_A):
            best_A, best_order = A2, shaken

    return best_A, best_order

def main() -> np.ndarray:
    K = 400
    seed = 42
    restarts = 64
    candidate_set, used_order = improve_by_order_shake(K, seed, restarts)
    indicators = np.zeros(K, dtype=int)
    for v in candidate_set:
        indicators[v - 1] = 1
    print(f"K={K}, size={len(candidate_set)}")
    return indicators

if __name__ == "__main__":
    indicators = main()

# Ensure compatibility with evaluators that expect a global variable
try:
    indicators  # type: ignore[name-defined]
except NameError:
    indicators = main()
