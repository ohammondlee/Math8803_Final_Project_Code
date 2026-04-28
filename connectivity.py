import torch
import numpy as np


def build_connectivity(config, shape):
    t = config["type"]
    if t == "low_rank":
        return low_rank(shape, config["rank"], config.get("base_std", 1.25), config.get("base_W0"))
    elif t == "spectral":
        return spectral_radius_init(shape, config["spectral_radius"])
    elif t == "orthogonal":
        return dist_to_orth_init(shape, config["orthogonal_distance"])
    elif t =="sparse":
        return sparse_init(shape, config["sparsity"])
    elif t == "random":
        return torch.randn(*shape)
    else:
        raise ValueError(f"Unknown connectivity type: {t}")

def low_rank(shape, rank, base_std=1.25, base_W0=None):
    n, m = shape
    assert n == m, "SVD truncated assumes square matrix"
    if base_W0 is not None:
        W0 = base_W0
    else:
        W0 = base_std * torch.randn(n, n) / (n ** 0.5)
    U, S, VT = torch.linalg.svd(W0)
    new_S = S.clone()
    new_S[rank:] = 0
    W_trunc = U @ torch.diag(new_S) @ VT
    # Renormalize to same norm as W0
    W_trunc = W_trunc * (torch.norm(W0) / torch.norm(W_trunc))
    return W_trunc

def spectral_radius_init(shape, rho):
    W = torch.randn(*shape)
    eigvals = torch.linalg.eigvals(W)
    current = eigvals.abs().max()
    return (W * (rho / current)).real

def orthogonal_init(shape, alpha, scale=1.0):
    n, m = shape
    k = min(n, m)
    U, _ = torch.linalg.qr(torch.randn(n, k))
    V, _ = torch.linalg.qr(torch.randn(m, k))
    # log-normal singular values centered at 1
    log_s = alpha * torch.randn(k)
    s = torch.exp(log_s)  # ensures positivity
    s = s / s.mean()
    S = torch.diag(s)
    return scale * (U @ S @ V.T).real


def dist_to_orth_init(shape, alpha, target_norm=None):
    """
    alpha = 0: Symmetric (Far from orthogonal)
    alpha = 1: Pure Orthogonal
    """
    W_sym = np.random.randn(*shape)
    W_sym = (W_sym + W_sym.T) / 2
    Q, _ = np.linalg.qr(np.random.randn(*shape))
    W = (1 - alpha) * (W_sym / np.linalg.norm(W_sym)) + alpha * (Q / np.linalg.norm(Q))
    if target_norm is not None:
        W = W / np.linalg.norm(W) * target_norm
        
    return W

def sparse_init(shape, sparsity, target_norm=None):
    W = np.random.randn(*shape)
    mask = np.random.rand(*shape) < sparsity
    W = W * mask
    if target_norm:
        W = W / np.linalg.norm(W) * target_norm
    return W