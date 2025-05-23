from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from misc import get_model_keys, model_config, print_log


def main():
    model_keys = get_model_keys(from_arg=True)

    root_dir = Path("../output")
    in_dir = root_dir / "merged"
    out_dir = root_dir / "rank"
    out_dir.mkdir(parents=True, exist_ok=True)

    def make_rank(stem):
        file = in_dir / f"{stem}.safetensors"

        if file.exists():
            print_log("working on", stem, "...")

            save_file(
                {
                    data_key: logits.argsort(dim=-1).argsort(dim=-1).to(torch.int8)
                    for data_key, logits in load_file(file).items()
                },
                out_dir / f"{stem}.safetensors",
            )

            print_log(stem, "finished.")

    with ProcessPoolExecutor(max_workers=4) as pool:
        for model_key in model_keys:
            for layer_idx in range(model_config.loc[model_key, "num_layers"]):
                pool.submit(make_rank, f"{model_key}_{layer_idx:02d}")


if __name__ == "__main__":
    main()
