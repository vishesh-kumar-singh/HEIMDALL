import numpy as np

def temporal_curvature_sequence(h: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Computes kappa_t for t = 2 .. T-1 (length T-2), where kappa[i] corresponds to step t = i+2.
    Formula: kappa_t = (v_t . v_{t-1}) / (||v_t|| * ||v_{t-1}||)
    Displacement: v_t = h_t - h_{t-1}
    Range: [-1, 1]
    """
    T = h.shape[0]
    if T < 3:
        return np.array([], dtype=np.float32)
    
    # v[i] = h[i+1] - h[i] -> v[0] = h_1 - h_0, v[1] = h_2 - h_1 (which is v_2)
    v = np.diff(h, axis=0) # shape (T-1, d)
    
    # Dot product between v_t and v_{t-1}
    num = np.sum(v[1:] * v[:-1], axis=1)
    
    # Product of norms
    norm_curr = np.linalg.norm(v[1:], axis=1)
    norm_prev = np.linalg.norm(v[:-1], axis=1)
    denom = (norm_curr * norm_prev) + eps
    
    kappa = np.divide(num, denom, out=np.full_like(num, np.nan), where=denom != 0)
    return np.clip(kappa, -1.0, 1.0)


def dispersion_sequence(h: np.ndarray) -> np.ndarray:
    """
    Computes dispersion V(lambda) for lambda = 1 .. T-1 (length T-1).
    Formula: V(lambda) = (1/lambda) * sum_{i=0}^lambda ||h_i - h_bar^(lambda)||^2
    where h_bar^(lambda) = mean(h_0 .. h_lambda)
    """
    T = h.shape[0]
    if T < 2:
        return np.array([], dtype=np.float32)
    
    V = np.zeros(T - 1, dtype=np.float32)
    for lam in range(1, T):
        prefix = h[:lam + 1] # h_0 .. h_lambda (length lambda + 1)
        mean = prefix.mean(axis=0, keepdims=True)
        sq_dists = np.sum((prefix - mean) ** 2, axis=-1)
        V[lam - 1] = np.sum(sq_dists) / lam
    return V


def variance_slope_sequence(V: np.ndarray) -> np.ndarray:
    """
    Computes beta_t = OLS slope of V(lambda) vs lambda over an expanding window
    lambda = 1 .. t for each t = 2 .. T-1 (length T-2, matching kappa_t).
    """
    T_minus_1 = len(V)
    if T_minus_1 < 2:
        return np.array([], dtype=np.float32)
    
    betas = []
    lambdas_full = np.arange(1, T_minus_1 + 1, dtype=np.float32)
    
    for t in range(2, T_minus_1 + 1):
        lam = lambdas_full[:t]
        v = V[:t]
        lam_bar, v_bar = lam.mean(), v.mean()
        num = np.sum((lam - lam_bar) * (v - v_bar))
        den = np.sum((lam - lam_bar) ** 2)
        betas.append(float(num / den) if den != 0 else np.nan)
        
    return np.array(betas, dtype=np.float32)


def run_unit_tests():
    """Unit tests on synthetic toy trajectories."""
    print("Running synthetic trajectory unit tests...")
    
    # 1. Straight line: h_t = [t, 2t, 3t] -> constant velocity v_t -> kappa should be 1.0 everywhere
    t_steps = 10
    h_line = np.array([[t, 2*t, 3*t] for t in range(t_steps)], dtype=np.float32)
    kappa_line = temporal_curvature_sequence(h_line)
    assert np.allclose(kappa_line, 1.0, atol=1e-5), f"Straight line failed: {kappa_line}"
    
    # 2. Perfect Zig-Zag: h_t reverses direction every step -> kappa should be -1.0
    h_zigzag = np.array([[0, 0], [1, 0], [0, 0], [1, 0], [0, 0]], dtype=np.float32)
    kappa_zigzag = temporal_curvature_sequence(h_zigzag)
    assert np.allclose(kappa_zigzag, -1.0, atol=1e-5), f"Zig-zag failed: {kappa_zigzag}"
    
    # 3. Variance slope on expanding dispersion
    V_test = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32) # slope = 1.0
    beta_test = variance_slope_sequence(V_test)
    assert np.allclose(beta_test, 1.0, atol=1e-5), f"Variance slope failed: {beta_test}"
    
    print("✅ All geometric unit tests passed successfully!")

if __name__ == "__main__":
    run_unit_tests()
