import torch
import neurogym as ngym
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def build_task(config):
    mode = config["task_mode"]
    if mode == "ngym":
        return NeuroGymTask(config)
    elif mode == "sMNIST":
        return SMNISTTask(config)
    else:
        raise ValueError(f"Unknown task_mode: {mode}")

class NeuroGymTask:
    def __init__(self, config):
        args_task = config["task"]
        t_mult = config.get("t_mult", 1)
        self.batch_size = config.get("batch_size", 32)
        if args_task == "2AF":
            task = "PerceptualDecisionMaking-v0"
            timing = {
                "fixation": 0 * t_mult,
                "stimulus": 700 * t_mult,
                "delay": 0 * t_mult,
                "decision": 100 * t_mult,
            }
            seq_len = 8 * t_mult
        elif args_task == "DMS":
            task = "DelayMatchSample-v0"
            seq_len = 8 * t_mult
            timing = {
                "fixation": 0 * t_mult,
                "sample": 100 * t_mult,
                "delay": 500 * t_mult,
                "test": 100 * t_mult,
                "decision": 100 * t_mult,
            }
        elif args_task == "CXT":
            task = "ContextDecisionMaking-v0"
            seq_len = 8 * t_mult
            timing = {
                "fixation": 0 * t_mult,
                "stimulus": 200 * t_mult,
                "delay": 500 * t_mult,
                "decision": 100 * t_mult,
            }
        else:
            raise ValueError(f"Unknown ngym task: {args_task}")

        kwargs = {"dt": 100, "timing": timing}
        self.dataset = ngym.Dataset(
            task,
            env_kwargs=kwargs,
            batch_size=self.batch_size,
            seq_len=seq_len,
        )
        self.env = self.dataset.env
        self.input_dim = self.env.observation_space.shape[0]
        self.output_dim = self.env.action_space.n
        self.dt = self.env.dt
        self.seq_len = seq_len

    def sample_batch(self, config):
        x, y = self.dataset()
        x = torch.tensor(x, dtype=torch.float32).to(config.get("device", "cpu"))
        y = torch.tensor(y, dtype=torch.long).to(config.get("device", "cpu"))
        return x, y
    
    def get_reference_batch(self, config, num_batches=10):
        inputs0 = []
        for _ in range(num_batches):
            x, _ = self.sample_batch(config)
            inputs0.append(x)
        inputs0 = torch.cat(inputs0, dim=1)
        return inputs0.float()
    
class SMNISTTask:
    def __init__(self, config):
        self.batch_size = config.get("batch_size", 200)
        self.seq_len = 28
        self.input_dim = 28
        self.output_dim = 10
        self.dt = None

        data = datasets.MNIST(
            "data", 
            train=True, 
            download=True, 
            transform=transforms.ToTensor()
        )
        train_set, val_set = random_split(data, [55000, 5000])
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size)
        self.val_loader = DataLoader(val_set, batch_size=self.batch_size)
        self.iter_train_loader = iter(self.train_loader)
        self.iter_val_loader = iter(self.val_loader)

    def sample_batch(self):
        x, y = next(self.iter_train_loader)
        x = x[:, 0].permute(1, 0, 2)
        return x, y
    
    def get_reference_batch(self):
        x, _ = self.sample_batch()
        return x.float()