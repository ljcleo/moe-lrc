from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from safetensors.torch import save_file

from misc import get_data_keys, get_model_keys, model_config, print_log


def process(in_dir, out_dir, data_keys, stem):
    if sum(1 for _ in in_dir.glob(f"{stem}_*.pt")) > 0:
        print_log("working on", stem, "...")

        save_file(
            {data_key: torch.load(in_dir / f"{stem}_{data_key}.pt") for data_key in data_keys},
            out_dir / f"{stem}.safetensors",
        )

        print_log(stem, "finished.")


def main():
    data_keys = get_data_keys()
    model_keys = get_model_keys(from_arg=True)

    root_dir = Path("../output")
    in_dir = root_dir / "raw"
    out_dir = root_dir / "merged"
    out_dir.mkdir(parents=True, exist_ok=True)

    futures = []

    with ProcessPoolExecutor(max_workers=4) as pool:
        for model_key in model_keys:
            for layer_idx in range(model_config.loc[model_key, "num_layers"]):
                futures.append(
                    pool.submit(process, in_dir, out_dir, data_keys, f"{model_key}_{layer_idx:02d}")
                )

    for future in futures:
        future.result()


if __name__ == "__main__":
    main()
