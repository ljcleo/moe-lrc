from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from misc import get_global_config, get_model_keys, model_config, print_log


def seg_stat(in_dir, out_dir, model_key):
    results = {}

    for layer_idx in range(model_config.loc[model_key, "num_layers"]):
        stem = f"{model_key}_{layer_idx:02d}"
        file = in_dir / f"{stem}.safetensors"

        if file.exists():
            print_log("working on", stem, "...")
            num_experts = model_config.loc[model_key, "num_experts"]
            top_k = model_config.loc[model_key, "top_k"]

            for data_key, rank in load_file(file).items():
                mask = (rank >= num_experts - top_k).to(torch.int16)
                seg_lens = get_global_config("seg_lengths")
                seg_len = 1

                while seg_len < seg_lens[-1] and seg_len < rank.shape[1]:
                    mask = mask[:, :-seg_len] + mask[:, seg_len:]
                    seg_len <<= 1

                    if seg_len in seg_lens:
                        cnt = mask.new_zeros(seg_len + 1, *mask.shape[1:])
                        cnt.scatter_add_(0, mask.long(), torch.ones_like(mask))
                        cnt.transpose_(0, 2)
                        results[f"{layer_idx:02d}_{data_key}_{seg_len:03d}"] = cnt.contiguous()

            print_log(stem, "finished.")

    if len(results) > 0:
        print_log("saving", model_key, "...")
        save_file(results, out_dir / f"{model_key}.safetensors")
        print_log(model_key, "saved.")


def main():
    model_keys = get_model_keys(from_arg=True)

    root_dir = Path("../output")
    in_dir = root_dir / "rank"
    out_dir = root_dir / "srp_pos"
    out_dir.mkdir(parents=True, exist_ok=True)

    futures = []

    with ProcessPoolExecutor(max_workers=2) as pool:
        for model_key in model_keys:
            futures.append(pool.submit(seg_stat, in_dir, out_dir, model_key))

    for future in futures:
        future.result()


if __name__ == "__main__":
    main()
