import torch
import numpy as np
import networkx as nx
import community as community_louvain

def weight_distance(W0, Wt):
    return torch.norm(W0 - Wt)

def spectral_radius(W):
    eigvals = torch.linalg.eigvals(W)
    return eigvals.abs().max().item()

def orthogonality_error(W):
    I = torch.eye(W.shape[0])
    return torch.norm(W.T @ W - I).item()

def representation_change(model, x, f0):
    with torch.no_grad():
        f_t = model(x)
    return torch.norm(f_t - f0).item()

def kernel_alignment(K0, Kf):
    return torch.sum(K0 * Kf) / (torch.norm(K0) * torch.norm(Kf))

def representation_similarity(H0, H):
    KR0 = H0[-1] @ H0[-1].T
    KR = H[-1] @ H[-1].T
    return torch.sum(KR * KR0) / (torch.norm(KR0) * torch.norm(KR))

def sign_similarity(H0, H):
    return (torch.sign(H0) == torch.sign(H)).float().mean()

def compute_ntk(model, inputs, task_mode):
    W = model.rnn.h2h.weight
    outputs, _, _ = model(inputs)
    grads = []
    if task_mode == "ngym":
        T, B, C = outputs.shape
        for t in range(T):
            for b in range(B):
                for k in range(C):
                    g = torch.autograd.grad(
                        outputs[t, b, k],
                        W,
                        retain_graph=True
                    )[0]
                    grads.append(g.flatten())
    elif task_mode == "sMNIST":
        B, C = outputs[-1].shape
        for b in range(B):
            for k in range(C):
                g = torch.autograd.grad(
                    outputs[-1, b, k],
                    W,
                    retain_graph=True
                )[0]
                grads.append(g.flatten())

    J = torch.stack(grads)
    K = J @ J.T
    return K

def compute_modularity_Q(W0, symmetrize=True, threshold_pct=0.15):
    """
    Compute modularity Q from an RNN weight matrix W0.
    """
    W = np.abs(W0.copy())
    if symmetrize:
        W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0)

    if threshold_pct < 1.0:
        thresh_val = np.percentile(W[W > 0], 100 * (1 - threshold_pct))
        W[W < thresh_val] = 0

    G = nx.from_numpy_array(W)
    Q_vals, partitions = [], []
    for _ in range(10):
        p = community_louvain.best_partition(G, weight='weight')
        q = community_louvain.modularity(p, G, weight='weight')
        Q_vals.append(q)
        partitions.append(p)
    best = np.argmax(Q_vals)
    return Q_vals[best], partitions[best]

def compute_functional_modularity(activity, threshold_pct=0.15):
    A = activity.reshape(-1, activity.shape[-1])

    std = A.std(axis=0)
    active_mask = std > 1e-8
    
    if active_mask.sum() < 2:
        return 0.0, {}
    
    A_active = A[:, active_mask]
    corr = np.corrcoef(A_active, rowvar=False)
    corr = np.nan_to_num(corr)

    # Keeping only positive correlations
    W_func = np.clip(corr, 0, None)
    np.fill_diagonal(W_func, 0)

    # Proportional thresholding (which seems to be standard)
    if threshold_pct < 1.0:
        thresh_val = np.percentile(W_func[W_func > 0], 
                                   100 * (1 - threshold_pct))
        W_func[W_func < thresh_val] = 0

    G = nx.from_numpy_array(W_func)
    Q_vals, partitions = [], []
    for _ in range(10):
        p = community_louvain.best_partition(G, weight='weight')
        q = community_louvain.modularity(p, G, weight='weight')
        Q_vals.append(q)
        partitions.append(p)
    best = np.argmax(Q_vals)
    return Q_vals[best], partitions[best]