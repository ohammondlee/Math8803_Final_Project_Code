from models import Net
from experiments import sweep_rank, sweep_spectral, sweep_orthogonal, sweep_sparse
from tasks import build_task
from utils import load_config, set_seed, save_results, update_save_file

def build_model(config):
    if config["model"] == "rnn":
        model = Net(config)
        return model.to(config.get("device", "cpu"))
    else:
        raise ValueError(f"Unknown model: {config['model']}")

def main():
    compiled_results={}
    config = load_config("configs.yaml")
    
    for i in range(config.get("sample_size",1)):

        print(f"now running trial {i}")

        set_seed(config.get("seed", 0)+i)
        task = build_task(config)

        config["input_dim"] = task.input_dim
        config["output_dim"] = task.output_dim
        config["seq_len"] = task.seq_len

        if task.dt is not None:
            config["dt"] = task.dt

        inputs0 = task.get_reference_batch(config)
        inputs0 = inputs0.to(config.get("device", "cpu"))
        exp_name = config["experiment"]

        if exp_name == "rank_sweep":
            results = sweep_rank(config, task, build_model, inputs0)
        elif exp_name == "spectral_sweep":
            results = sweep_spectral(config, task, build_model, inputs0)
        elif exp_name == "orthogonal_sweep":
            results = sweep_orthogonal(config, task, build_model, inputs0)
        elif exp_name == "sparse_sweep":
            results = sweep_sparse(config, task, build_model, inputs0)
        else:
            raise ValueError(f"Unknown experiment: {exp_name}")
        
        update_save_file(results, save_dir="results", filename=f"{exp_name}_{config['task_mode']}_{config['task']}_{config['additional_details']}.json")

        # # ---- aggregation step ----
        # for param, metrics in results.items():
        #     if param not in compiled_results:
        #         compiled_results[param] = {m: [] for m in metrics}

        #     for metric_name, value in metrics.items():
        #         compiled_results[param][metric_name].append(value)

        mode = config["task_mode"]
        task_name = config["task"]
    # save_results(
    #     compiled_results,
    #     save_dir="results",
    #     filename=f"{exp_name}_{mode}_{task_name}_{config['additional_details']}.json"
    # )

if __name__ == "__main__":
    main()