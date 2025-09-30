from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from misc import get_global_config, get_model_keys, model_config, print_log


def seg_stat(bs_size, in_dir, out_dir, model_key):
    rng = torch.manual_seed(hash(model_key))
    bs_w = [None]

    def calc_stat(mask: torch.Tensor, seg_len: int):
        cnt = mask.new_zeros(seg_len + 1, *mask.shape[1:])
        cnt.scatter_add_(0, mask.long(), torch.ones_like(mask))
        cnt = cnt.permute(2, 0, 1).to(torch.int32)

        if bs_w[0] is None:
            bs_len = mask.shape[1]
            bs_idx = torch.randint(bs_len, (bs_len, bs_size), generator=rng)
            bs_weight = cnt.new_zeros(bs_idx.shape)
            bs_weight.scatter_add_(0, bs_idx, torch.ones_like(bs_weight))
            bs_w[0] = torch.cat([bs_weight.new_ones(bs_len, 1), bs_weight], dim=-1)

        seg_stat = (cnt @ bs_w[0]).permute(2, 0, 1)

        num_matches = (
            (seg_stat * torch.arange(seg_len + 1, dtype=torch.int32))
            .flip(dims=(-1,))
            .cumsum(dim=-1, dtype=torch.int32)
            .flip(dims=(-1,))
        ).contiguous()

        num_preds = (
            seg_stat.flip(dims=(-1,)).cumsum(dim=-1, dtype=torch.int32).flip(dims=(-1,)) * seg_len
        ).contiguous()

        num_acts = num_matches[..., :1].contiguous()
        return num_matches, num_preds, num_acts

    num_layers = model_config.loc[model_key, "num_layers"]
    num_experts = model_config.loc[model_key, "num_experts"]
    top_k = model_config.loc[model_key, "top_k"]
    results = {}

    for layer_idx in range(num_layers):
        stem = f"{model_key}_{layer_idx:02d}"
        file = in_dir / f"{stem}.safetensors"

        if file.exists():
            print_log("working on", stem, "...")

            for data_key, rank in load_file(file).items():
                mask = (rank >= num_experts - top_k).to(torch.int16).transpose(0, 1)
                seg_lens = get_global_config("seg_lengths")
                seg_len = 1

                while seg_len < seg_lens[-1] and seg_len < rank.shape[0]:
                    mask = mask[:-seg_len] + mask[seg_len:]
                    seg_len <<= 1

                    if seg_len in seg_lens:
                        key = f"{layer_idx:02d}_{data_key}_{seg_len:03d}"
                        print_log(key)

                        (results[f"{key}_m"], results[f"{key}_p"], results[f"{key}_a"]) = calc_stat(
                            mask, seg_len
                        )

            print_log(stem, "finished.")

    if len(results) > 0:
        print_log("saving", model_key, "...")
        save_file(results, out_dir / f"{model_key}.safetensors")
        print_log(model_key, "saved.")


def main():
    model_keys = get_model_keys(from_arg=True)
    bs_size: int = get_global_config("num_bootstrap")

    root_dir = Path("../output")
    in_dir = root_dir / "rank"
    out_dir = root_dir / "srp"
    out_dir.mkdir(parents=True, exist_ok=True)

    futures = []

    with ProcessPoolExecutor(max_workers=2) as pool:
        for model_key in model_keys:
            futures.append(pool.submit(seg_stat, bs_size, in_dir, out_dir, model_key))

    for future in futures:
        future.result()


if __name__ == "__main__":
    main()
