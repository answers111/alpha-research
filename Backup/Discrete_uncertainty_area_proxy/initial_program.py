import numpy as np

def evaluate_uncertainty_area_simple(f: np.ndarray, tau: float = 1e-3):
    """
    product = A_tau(f) * A_{N*tau}(FFT(f)),  MINIMIZE
    Returns a dict with keys: product, Af, AF, k_time, k_freq, N, tau
    """
    f = np.asarray(f)
    N = f.size
    if N == 0:
        return {"product": float("inf"), "Af": 0.0, "AF": 0.0,
                "k_time": 0, "k_freq": 0, "N": 0, "tau": float(tau)}

    F = np.fft.fft(f)                  # default NumPy scaling (forward unscaled)
    mask_time = np.abs(f) > tau
    mask_freq = np.abs(F) > (tau * N)  # align threshold to FFT magnitude scale

    k_time = int(mask_time.sum())
    k_freq = int(mask_freq.sum())
    Af = k_time / N
    AF = k_freq / N
    product = Af * AF

    return {
        "product": float(product),
        "Af": float(Af),
        "AF": float(AF),
        "k_time": k_time,
        "k_freq": k_freq,
        "N": int(N),
        "tau": float(tau),
    }

def make_dirac_comb(N: int):
    s = int(np.sqrt(N)) or 1
    x = np.zeros(N, dtype=float)
    x[::s] = 1.0
    return x

def main():
    N = 256
    tau = 1e-3
    x = make_dirac_comb(N)
    # Optionally print product for info, but return x for evaluator consumption
    res = evaluate_uncertainty_area_simple(x, tau)
    print(res["product"])
    return x

if __name__ == "__main__":
    x = main()
