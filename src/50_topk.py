from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from misc import get_model_keys, model_config, print_log


def make_rank(in_dir, out_dir, model_key, layer_idx):
    stem = f"{model_key}_{layer_idx:02d}"
    file = in_dir / f"{stem}.safetensors"

    if file.exists():
        print_log("working on", stem, "...")
        top_k = model_config.loc[model_key, "top_k"]

        save_file(
            {
                data_key: logits.topk(top_k, dim=-1).indices.to(torch.int8)
                for data_key, logits in load_file(file).items()
            },
            out_dir / f"{stem}.safetensors",
        )

        print_log(stem, "finished.")


def main():
    model_keys = get_model_keys(from_arg=True)

    root_dir = Path("../output")
    in_dir = root_dir / "rank"
    out_dir = root_dir / "topk"
    out_dir.mkdir(parents=True, exist_ok=True)

    futures = []

    with ProcessPoolExecutor(max_workers=4) as pool:
        for model_key in model_keys:
            for layer_idx in range(model_config.loc[model_key, "num_layers"]):
                futures.append(pool.submit(make_rank, in_dir, out_dir, model_key, layer_idx))

    for future in futures:
        future.result()


if __name__ == "__main__":
    main()
