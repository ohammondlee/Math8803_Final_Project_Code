from copy import deepcopy
from train import train
from metrics import compute_ntk, representation_similarity, sign_similarity, kernel_alignment, orthogonality_error, spectral_radius, compute_modularity_Q, compute_functional_modularity
from tqdm import tqdm
import torch
import torch.nn as nn

def run_single_experiment(config, task, model_builder, inputs0):
    model = model_builder(config)
    W0 = model.rnn.h2h.weight.clone()
    _, activity0, _ = model(inputs0)
    K0 = compute_ntk(model, inputs0, config["task_mode"])
    logs = train(model, task, config)
    _, activity, _ = model(inputs0)
    Kf = compute_ntk(model, inputs0, config["task_mode"])
    Wf = model.rnn.h2h.weight.clone()
    
    # Compute modularity metrics
    Q0, _ = compute_modularity_Q(W0.detach().cpu().numpy())
    Qf, _ = compute_modularity_Q(Wf.detach().cpu().numpy())
    func_mod0, _ = compute_functional_modularity(activity0.detach().cpu().numpy())
    func_modf, _ = compute_functional_modularity(activity.detach().cpu().numpy())
    
    results = {
        "weight_dist": logs[-1]["weight_dist"],
        "rep_sim": representation_similarity(activity0, activity),
        "sign_sim": sign_similarity(activity0, activity),
        "kernel_alignment": kernel_alignment(K0, Kf),
        # "orthogonal_distance": orthogonality_error(activity0),
        # "spectral_radius": spectral_radius(activity0),
        "loss": logs[-1]["loss"],
        "initial_modularity_Q": Q0,
        "final_modularity_Q": Qf,
        "modularity_Q_change": Qf - Q0,
        "initial_functional_modularity": func_mod0,
        "final_functional_modularity": func_modf,
        "functional_modularity_change": func_modf - func_mod0,
    }
    print(results)
    return results

def run_with_lr_sweep(config, task, model_builder, inputs0):
    best_result = None
    best_loss = float("inf")
    for lr in config["lr_list"]:
        cfg = config.copy()
        cfg["lr"] = float(lr)
        result = run_single_experiment(cfg, task, model_builder, inputs0)
        if result["loss"] < best_loss:
            best_loss = result["loss"]
            best_result = result
    return best_result

def sweep_rank(config, task, model_builder, inputs0):
    base_W0 = config["connectivity"].get("base_std", 1.25) * torch.randn(config["hidden_dim"], config["hidden_dim"]) / (config["hidden_dim"] ** 0.5)
    results = {}
    for r in tqdm(config["rank_list"], desc="Rank Sweep", position=0):
        cfg = deepcopy(config)
        cfg["connectivity"]["type"] = "low_rank"
        cfg["connectivity"]["rank"] = r
        cfg["connectivity"]["base_W0"] = base_W0
        result = run_with_lr_sweep(cfg, task, model_builder, inputs0)
        results[f"rank_{r}"] = result
    return results

def sweep_spectral(config, task, model_builder, inputs0):
    results = {}
    for rho in tqdm(config["radius_list"], desc="Spectral Sweep", position=0):
        cfg = deepcopy(config)
        cfg["connectivity"]["type"] = "spectral"
        cfg["connectivity"]["spectral_radius"] = rho
        result = run_with_lr_sweep(cfg, task, model_builder, inputs0)
        results[f"radius_{rho}"] = result
    return results

def sweep_orthogonal(config, task, model_builder, inputs0):
    results = {}
    for rho in tqdm(config["ortho_list"], desc="Orthogonal Sweep", position=0):
        cfg = deepcopy(config)
        cfg["connectivity"]["type"] = "orthogonal"
        cfg["connectivity"]["orthogonal_distance"] = rho
        result = run_with_lr_sweep(cfg, task, model_builder, inputs0)
        results[f"orthogonal_distance_{rho}"] = result
    return results

def sweep_sparse(config, task, model_builder, inputs0):
    results = {}
    for rho in tqdm(config["sparse_list"], desc="Sparse Sweep", position=0):
        cfg = deepcopy(config)
        cfg["connectivity"]["type"] = "sparse"
        cfg["connectivity"]["sparsity"] = rho
        result = run_with_lr_sweep(cfg, task, model_builder, inputs0)
        results[f"sparsity_{rho}"] = result
    return results