import argparse
import yaml
import subprocess
import torch


def run_command(command):
    print(f"\nRunning:\n{' '.join(command)}\n")
    subprocess.run(command, check=True)


def build_command(script, args_dict):
    cmd = ["python", script]
    for key, val in args_dict.items():
        if val is None:
            continue
        if isinstance(val, bool):
            if val:
                cmd.append(f"--{key}")
        elif isinstance(val, list):
            for item in val:
                cmd.append(f"--{key}")
                cmd.append(str(item))
        else:
            cmd.append(f"--{key}")
            cmd.append(str(val))
    return cmd


def main(config_path, steps):
    torch.cuda.empty_cache()

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "hnmine" in steps:  # Mine the hard negative passages for each training sample
        hn_args = config["hn_mine"]
        hn_cmd = build_command("models/FlagEmbedding/scripts/hn_mine.py", hn_args)
        print(hn_cmd)
        run_command(hn_cmd)

    if "score" in steps: # Obtain Positive and Negative scores to fine tune using Knowledge Destillation
        score_args = config["score_ranking"]
        score_cmd = build_command("models/FlagEmbedding/scripts/add_reranker_score.py", score_args)
        print(score_cmd)
        run_command(score_cmd)

    if "train" in steps:   # Model fine tunning
        train_args = config["fine_tune"]
        script_name = train_args.pop("script")
        train_cmd = ["torchrun", "--nproc_per_node", str(config["fine_tune"].pop("num_devices")), "-m", script_name]
        for key, val in train_args.items():
            if val is None:
                continue
            if isinstance(val, bool):
                if val:
                    train_cmd.append(f"--{key}")
            elif isinstance(val, list):
                for item in val:
                    train_cmd.append(f"--{key}")
                    train_cmd.append(str(item))
            else:
                train_cmd.append(f"--{key}")
                train_cmd.append(str(val))
        print(train_cmd)
        run_command(train_cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--steps", nargs="+", choices=["hnmine", "score", "train"], default=["hnmine", "score", "train"],
                        help="Pipeline steps to run (choose from: hnmine, score, train)")
    args = parser.parse_args()
    main(args.config, args.steps)
