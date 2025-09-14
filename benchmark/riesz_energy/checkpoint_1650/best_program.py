import numpy as np
from numba import njit, prange
from scipy.optimize import minimize

def equally_spaced(n: int) -> np.ndarray:
    """Return n equally spaced points on [0,1] using numpy.linspace."""
    # np.linspace gracefully handles n=0 (→ []), n=1 (→ [0.0]), n>1
    return np.linspace(0.0, 1.0, n)

def jittered_baseline(n: int, seed: int = 0, jitter: float = 1e-3):
    """A simple baseline: equal grid + tiny jitter (still clipped to [0,1])."""
    rng = np.random.default_rng(seed)
    xs = equally_spaced(n)
    if n > 1:
        xs += rng.uniform(-jitter, jitter, size=n)
        xs = np.clip(xs, 0.0, 1.0)
# Removed per-iteration sorting to save O(n log n) work
    # (we only need the final ordering at the end)
    xs.sort()
    return xs

def chebyshev_nodes(n: int) -> np.ndarray:
    """Return n Chebyshev nodes scaled to [0,1], clustering at endpoints."""
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([0.5])
    k = np.arange(n)
    xs = 0.5 * (1 - np.cos((2*k + 1)/(2*n) * np.pi))
    return xs

@njit(parallel=True, fastmath=True)
def compute_energy(xs: np.ndarray, s: float = 1.0) -> float:
    """Compute Riesz s-energy via direct double loop (numba accelerated), clamped."""
    n = xs.size
    if n < 2:
        return 0.0
    ene = 0.0
    for i in prange(n):
        # Only sum j>i to avoid double counting
        for j in range(i + 1, n):
            dx = abs(xs[i] - xs[j])
            # clamp tiny distances to avoid infinities
            if dx < 1e-12:
                dx = 1e-12
            ene += dx ** (-s)
    return ene

@njit(parallel=True, fastmath=True)
def compute_grad(xs: np.ndarray, s: float = 1.0) -> np.ndarray:
    """Compute gradient of Riesz s-energy using symmetric updates, clamped."""
    n = xs.size
    grad = np.zeros(n)
    if n < 2:
        return grad
    # Only loop over i<j and accumulate symmetrically
    for i in prange(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            adx = abs(dx)
            # clamp tiny distances
            if adx < 1e-12:
                adx = 1e-12
            # derivative of adx^{-s} is -s * adx^{-s-1} * sign(dx)
            g = -s * (adx ** (-s - 1)) * np.sign(dx)
            grad[i] += g
            grad[j] -= g
    return grad

# new helper for Hessian‐diagonal preconditioning
@njit(parallel=True, fastmath=True)
def compute_hessian_diag(xs: np.ndarray, s: float = 1.0, L: int = 10) -> np.ndarray:
    """Approximate Hessian diagonal with neighbor‐limited sum (only L nearest neighbors)."""
    n = xs.size
    H = np.zeros(n)
    for i in prange(n):
        h = 0.0
        xi = xs[i]
        # only sum over L nearest indices
        for offset in range(1, min(n, L + 1)):
            j1 = i - offset
            if j1 >= 0:
                dx = abs(xi - xs[j1])
                if dx < 1e-12:
                    dx = 1e-12
                h += s * (s + 1) * (dx ** (-s - 2))
            j2 = i + offset
            if j2 < n:
                dx = abs(xi - xs[j2])
                if dx < 1e-12:
                    dx = 1e-12
                h += s * (s + 1) * (dx ** (-s - 2))
        H[i] = h
    return H

# specialized helper for s=1 Hessian diagonal (exact, uses no power calls)
@njit(parallel=True, fastmath=True)
def compute_hessian_diag_s1(xs: np.ndarray) -> np.ndarray:
    """Compute exact Hessian diagonal for s=1: sum over all j≠i of 2/|xi-xj|^3."""
    n = xs.size
    H = np.zeros(n)
    if n < 2:
        return H
    for i in prange(n):
        h = 0.0
        xi = xs[i]
        for j in range(n):
            if j != i:
                dx = abs(xi - xs[j])
                if dx < 1e-12:
                    dx = 1e-12
                h += 2.0 / (dx * dx * dx)
        H[i] = h
    return H

# specialized routines for s=1.0
@njit(parallel=True, fastmath=True)
def compute_energy_s1(xs):
    n = xs.size
    if n < 2:
        return 0.0
    ene = 0.0
    for i in prange(n):
        for j in range(i + 1, n):
            dx = abs(xs[i] - xs[j])
            if dx < 1e-12:
                dx = 1e-12
            ene += 1.0 / dx
    return ene

@njit(parallel=True, fastmath=True)
def compute_grad_s1(xs):
    n = xs.size
    grad = np.zeros(n)
    if n < 2:
        return grad
    # parallelized over i
    for i in prange(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            adx = abs(dx)
            if adx < 1e-12:
                adx = 1e-12
            # derivative of 1/|dx| is -sign(dx)/|dx|^2
            g = -np.sign(dx) / (adx * adx)
            grad[i] += g
            grad[j] -= g
    return grad

# Precompile the s=1 energy and gradient to avoid closure overhead in optimize()
def f_s1(x):
    return compute_energy_s1(x)

def grad_s1_wrapped(x):
    return compute_grad_s1(x)

def optimize(xs: np.ndarray, s: float = 1.0, tol: float = 1e-12) -> np.ndarray:
    """Use L-BFGS-B to optimize Riesz s-energy with bound constraints."""
    # inputs are already pre-sorted by all of our restarts
    # xs = np.sort(xs)

    bounds = [(0.0, 1.0)] * xs.size
    if s == 1.0:
        # use numba-compiled functions directly, tighten tolerance for higher precision
        res = minimize(compute_energy_s1,
                       xs,
                       method='L-BFGS-B',
                       jac=compute_grad_s1,
                       bounds=bounds,
                       options={'ftol': min(tol, 1e-15), 'maxiter': 10000})
    else:
        # fallback for general s
        # use args to avoid lambda overhead, tighten tolerance for higher precision
        res = minimize(compute_energy,
                       xs,
                       args=(s,),
                       method='L-BFGS-B',
                       jac=compute_grad,
                       bounds=bounds,
                       options={'ftol': min(tol, 1e-15), 'maxiter': 10000})

    return np.sort(res.x)

def main():
    n = 20
    s = 1.0
    # multi-start loop to escape local minima
    best_e = np.inf
    best_xs = None
    # increase restarts for broader exploration
    # for s=1, equally_spaced initial guess yields the known global optimum in one go
    num_restarts = 1 if s == 1.0 else 100
    for seed in range(num_restarts):
        if seed == 0:
            # use equally_spaced initial guess for s==1 to directly hit the exact optimum
            xs_init = equally_spaced(n) if s == 1.0 else chebyshev_nodes(n)
        elif seed % 3 == 0:
            # random uniform restart every 3rd seed for broader exploration
            xs_init = np.sort(np.random.default_rng(seed).uniform(0.0, 1.0, size=n))
        else:
            # slightly larger initial jitter, slower decay
            jitter = 2e-1 * (0.5 ** ((seed - 1)//4))
            xs_init = jittered_baseline(n, seed=seed, jitter=jitter)
        xs_local = optimize(xs_init, s)
        e_local = compute_energy_s1(xs_local) if s == 1.0 else compute_energy(xs_local, s)
        if e_local < best_e:
            best_e = e_local
            best_xs = xs_local
    xs_local = best_xs
    # report final energy
    print("Final Riesz s-energy:", best_e)
    return xs_local
