from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from safetensors.torch import save_file

from misc import get_data_keys, get_model_keys, model_config, print_log


def main():
    data_keys = get_data_keys()
    model_keys = get_model_keys(from_arg=True)

    root_dir = Path("../case")
    in_dir = root_dir / "raw"
    out_dir = root_dir / "merged"
    out_dir.mkdir(parents=True, exist_ok=True)

    def process(model_key):
        print_log("working on", model_key, "...")
        results = {}

        for layer_idx in range(model_config.loc[model_key, "num_layers"]):
            stem = f"{model_key}_{layer_idx:02d}"

            if sum(1 for _ in in_dir.glob(f"{stem}_*.pt")) > 0:
                results.update(
                    {
                        f"{layer_idx:02d}_{data_key}": torch.load(
                            in_dir / f"{stem}_{data_key}.pt"
                        ).squeeze(0)
                        for data_key in data_keys
                    }
                )

        save_file(results, out_dir / f"{model_key}.safetensors")

    futures = []

    with ThreadPoolExecutor(max_workers=20) as pool:
        for model_key in model_keys:
            futures.append(pool.submit(process, model_key))

    for future in futures:
        future.result()


if __name__ == "__main__":
    main()
