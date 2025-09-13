import numpy as np

def _normalize_rows(P):
    nrm = np.linalg.norm(P, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    return P / nrm

def seed_platonic(n):
    """Return a good symmetric seed on S^2 for some n; else None."""
    if n == 2:   # antipodal
        return np.array([[0,0,1],[0,0,-1]], dtype=float)
    if n == 3:   # equilateral on equator
        ang = 2*np.pi/3
        return np.array([[1,0,0],[np.cos(ang),np.sin(ang),0],[np.cos(2*ang),np.sin(2*ang),0]], dtype=float)
    if n == 4:   # tetrahedron
        return _normalize_rows(np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]], dtype=float))
    if n == 6:   # octahedron
        return np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=float)
    if n == 8:   # cube vertices
        V = np.array([[sx,sy,sz] for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)], dtype=float)
        return _normalize_rows(V)
    if n == 12:  # icosahedron (one realization)
        phi = (1+np.sqrt(5))/2
        V = []
        for s in (-1,1):
            V += [[0, s,  phi],[0, s, -phi],[ s,  phi,0],[ s, -phi,0],[ phi,0, s],[-phi,0, s]]
        V = np.array(V, dtype=float)
        return _normalize_rows(V)
    return None

def farthest_point_greedy(n, seed=None, rng=np.random.default_rng(0)):
    """Greedy max–min on S^2: start from seed (if any), then add points that maximize min angle."""
    def random_unit(k):
        X = rng.normal(size=(k,3)); return _normalize_rows(X)

    if seed is None:
        P = random_unit(1)   # start with one random point
    else:
        P = _normalize_rows(seed)
    while len(P) < n:
        # generate candidates and pick the one with largest min angle to current set
        C = random_unit(2000)  # candidates per iteration (tune as needed)
        # cosines to existing points
        cos = C @ P.T
        # min angle to set -> maximize this
        min_ang = np.arccos(np.clip(np.max(cos, axis=1), -1.0, 1.0))
        idx = np.argmax(min_ang)
        P = np.vstack([P, C[idx:idx+1]])
    return P

def main():
    n = 30
    seed = seed_platonic(n)
    pts = farthest_point_greedy(n, seed=seed, rng=np.random.default_rng(42))
    print(f"n={n}, points={len(pts)}")
    return pts

if __name__ == "__main__":
    points = main()
    # 保存为 npy 文件
    np.save("points.npy", points)
    print("已保存 points.npy")

# Ensure compatibility with evaluators that expect a global variable
try:
    points  # type: ignore[name-defined]
except NameError:
    points = main()
