import yaml
import torch
import random
import numpy as np
import os
import json

def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def save_config(config, path):
    with open(path, "w") as f:
        yaml.dump(config, f)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# For saving results

def tensor_to_python(obj):
    """Recursively convert tensors to python numbers/lists for JSON saving."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_python(v) for v in obj]
    else:
        return obj

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_results(results, save_dir="results", filename="results.json"):
    results = tensor_to_python(results)
    ensure_dir(save_dir)
    path = os.path.join(save_dir, filename)

    with open(path, "w") as f:
        json.dump(results, f, indent=4)

def save_tensor(tensor, path):
    ensure_dir(os.path.dirname(path))
    torch.save(tensor, path)

def load_tensor(path):
    return torch.load(path)

# Model helpers

def get_flat_params(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


import json
import os
import tempfile

def update_save_file(new_results, save_dir="results", filename="results.json"):
    new_results = tensor_to_python(new_results)
    ensure_dir(save_dir)
    path = os.path.join(save_dir, filename)

    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            existing = {}
    else:
        existing = {}

    for rank_key, metrics in new_results.items():
        if rank_key not in existing:
            existing[rank_key] = {}
        if metrics is None:
            continue
        for metric, values in metrics.items():
            if values is None:
                continue
            if metric not in existing[rank_key]:
                existing[rank_key][metric] = []

            if isinstance(values, list):
                existing[rank_key][metric].extend(values)
            else:
                existing[rank_key][metric].append(values)

    dir_name = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile("w", delete=False, dir=dir_name) as tmp:
        json.dump(existing, tmp, indent=4)
        temp_name = tmp.name

    os.replace(temp_name, path)

def load_save_file(save_dir="results", filename="results.json"):
    path = os.path.join(save_dir, filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)

def load_metrics(results):
    out={}
    for key in results.keys():
        out[key] = {}
        for metric in results[key].keys():
            if metric in ["w0","w1"]:
                continue
            out[key][metric] = results[key][metric]
    return out

