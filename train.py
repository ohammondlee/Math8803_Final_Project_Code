import torch
from tqdm import trange
import numpy as np
from metrics import weight_distance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_recurrent_weights(model):
    return model.rnn.h2h.weight.detach().flatten()

def train(model, dataloader, config):
    opt = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
    W0 = get_recurrent_weights(model).clone()
    logs = []
    for step in trange(config["n_iter"], desc="Training", position=1, leave=False):
        x, y = dataloader.sample_batch(config)
        pred, _, _ = model(x)
        loss = compute_loss(pred, y, config)

        opt.zero_grad()
        loss.backward()
        opt.step()

        Wt = get_recurrent_weights(model) 
        if step % 1000 == 0:
            Wt = get_recurrent_weights(model)
            logs.append({
                "weight_dist": weight_distance(W0, Wt).item(),
                "loss": loss.item()
            })
    
    Wt = get_recurrent_weights(model)
    logs.append({
        "weight_dist": weight_distance(W0, Wt).item(),
        "loss": loss.item()
    })

    return logs

def compute_loss(pred, y, config):
    if config["task_mode"] == "ngym":
        B = pred.shape[1]
        C = pred.shape[2]

        return torch.nn.functional.cross_entropy(
            pred.view(-1, C),
            y.view(-1)
        )
    elif config["task_mode"] == "sMNIST":
        return torch.nn.functional.cross_entropy(pred[-1], y)
    else:
        raise ValueError
