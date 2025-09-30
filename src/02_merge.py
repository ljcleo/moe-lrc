from pathlib import Path

import torch
from safetensors.torch import save_file

from misc import get_data_keys, get_model_keys


def main():
    data_keys = get_data_keys()
    model_keys = get_model_keys(from_arg=True)

    root_dir = Path("../data")
    in_dir = root_dir / "tokenized"
    out_dir = root_dir / "merged"
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_key in model_keys:
        print("working on", model_key, "...")

        save_file(
            {data_key: torch.load(in_dir / f"{model_key}_{data_key}.pt") for data_key in data_keys},
            out_dir / f"{model_key}.safetensors",
        )


if __name__ == "__main__":
    main()
