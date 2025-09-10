
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autoconvolution Peak Minimization
=================================

This program generates step heights for a probability density function
that minimizes the maximum value of its autoconvolution.
"""

import numpy as np
from typing import Dict

def evaluate_C1_upper_std(step_heights: np.ndarray) -> Dict[str, float]:
    """
    Standard-normalized C1 (support [-1/2,1/2], dx=1/K).
    - Project to feasible set: h >= 0 and ∫f = 1 (L1 normalization).
    - Objective: mu_inf = max_t (f*f)(t) (smaller is better).
    Returns: {"valid", "mu_inf", "ratio"(=mu_inf), "integral"(=1.0), "K"}
    """
    h = np.asarray(step_heights, dtype=float)
    if h.size == 0 or np.any(h < 0):
        return {"valid": 0.0, "mu_inf": float("inf"), "ratio": float("inf")}
    K = int(len(h))
    dx = 1.0 / K

    integral = float(np.sum(h) * dx)
    if integral <= 0:
        return {"valid": 0.0, "mu_inf": float("inf"), "ratio": float("inf")}
    h = h / integral  # ∫f = 1

    F = np.fft.fft(h, 2*K - 1)          # linear autoconvolution via padding
    conv = np.fft.ifft(F * F).real
    conv = np.maximum(conv, 0.0)        # clamp tiny negatives

    mu_inf = float(np.max(conv) * dx)
    return {"valid": 1.0, "mu_inf": mu_inf, "ratio": mu_inf, "integral": 1.0, "K": float(K)}

def make_candidate(K: int, kind: str = "cos2") -> np.ndarray:
    """
    Simple candidate builder on [-1/2,1/2] (NOT normalized here).
    
    Args:
        K: Number of discretization points
        kind: Type of candidate function ("box", "triangle", "cos2", "gauss")
    
    Returns:
        Step heights array
    """
    x = np.linspace(-1.0, 1.0, K)
    if kind == "box":
        h = np.ones(K)
    elif kind == "triangle":
        h = 1.0 - np.abs(x)
        h[h < 0] = 0.0
    elif kind == "cos2":
        h = np.cos(np.pi * x / 2.0) ** 2
    elif kind == "gauss":
        h = np.exp(-4.0 * x**2)
    else:
        raise ValueError(f"unknown kind={kind}")
    return h

def main():
    """
    Main function that generates step heights for autoconvolution minimization.
    
    Returns:
        numpy.ndarray: Step heights array
    """
    K = 128
    kind = "cos2"  # Change this to try different candidates (box/triangle/cos2/gauss)
    step_heights = make_candidate(K, kind)
    
    # Evaluate the result to verify it's valid
    result = evaluate_C1_upper_std(step_heights)
    print(f"Generated {kind} candidate with K={K}, mu_inf={result['mu_inf']:.6f}")
    
    return step_heights

if __name__ == "__main__":
    step_heights = main()

# Ensure compatibility with evaluators that expect a global variable
try:
    step_heights  # type: ignore[name-defined]
except NameError:
    step_heights = main()
