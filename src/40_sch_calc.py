from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from misc import get_global_config, get_model_keys, model_config, print_log


def seg_stat(bs_size, ci_confidence, in_dir, out_dir, model_key):
    rng = torch.manual_seed(hash(model_key))
    bs_lp = round(bs_size * ((1 - ci_confidence) / 2))
    bs_up = bs_size - bs_lp

    def calc_stat(mask: torch.Tensor, top_k: int, seg_len: int):
        sorted_val = mask.sort(dim=-1, descending=True).values.sum(dim=1)
        cnt = top_k * seg_len * mask.shape[1]
        res = sorted_val.sum(dim=0).cumsum(dim=-1) / (mask.shape[0] * cnt)

        bs_len = mask.shape[0]
        bs_idx = torch.randint(bs_len, (bs_size * bs_len,), generator=rng)
        bs_val = sorted_val[bs_idx].unflatten(0, (bs_size, bs_len))
        bs_res = (bs_val.sum(dim=1).cumsum(dim=-1) / (bs_val.shape[1] * cnt)).sort(dim=0).values

        res_l, res_u = res * 2 - bs_res[[bs_up, bs_lp]]
        return torch.stack([res_l, res, res_u]).contiguous()

    is_seq2seq = model_config.loc[model_key, "type"] == "seq2seq"
    num_layers = model_config.loc[model_key, "num_layers"]
    num_experts = model_config.loc[model_key, "num_experts"]
    top_k = model_config.loc[model_key, "top_k"]
    results: dict[str, torch.Tensor] = {}
    buf: dict[tuple[str, int], list[torch.Tensor]] = {}

    for layer_idx in range(num_layers):
        stem = f"{model_key}_{layer_idx:02d}"
        file = in_dir / f"{stem}.safetensors"

        if file.exists():
            print_log("working on", stem, "...")
            stage = "e" if is_seq2seq and layer_idx < num_layers // 2 else "d"

            for data_key, rank in load_file(file).items():
                mask = (rank >= num_experts - top_k).to(torch.int16)
                seg_lens = get_global_config("seg_lengths")
                seg_len = 1

                while seg_len < seg_lens[-1] and seg_len < rank.shape[1]:
                    mask = mask[:, :-seg_len] + mask[:, seg_len:]
                    seg_len <<= 1

                    if seg_len in seg_lens:
                        buf_key = f"{stage}_{data_key}", seg_len

                        if buf_key not in buf:
                            buf[buf_key] = [mask]
                        else:
                            buf[buf_key].append(mask)

                        results[f"{layer_idx:02d}_{data_key}_{seg_len:03d}"] = calc_stat(
                            mask, top_k, seg_len
                        )

            print_log(stem, "finished.")

    results.update(
        {
            f"{key}_{seg_len:03d}": calc_stat(
                torch.cat(values, dim=-1), top_k * len(values), seg_len
            )
            for (key, seg_len), values in buf.items()
        }
    )

    if len(results) > 0:
        print_log("saving", model_key, "...")
        save_file(results, out_dir / f"{model_key}.safetensors")
        print_log(model_key, "saved.")


def main():
    model_keys = get_model_keys(from_arg=True)
    bs_size: int = get_global_config("num_bootstrap")
    ci_confidence: float = get_global_config("ci_confidence")

    root_dir = Path("../output")
    in_dir = root_dir / "rank"
    out_dir = root_dir / "sch"
    out_dir.mkdir(parents=True, exist_ok=True)

    futures = []

    with ProcessPoolExecutor(max_workers=2) as pool:
        for model_key in model_keys:
            futures.append(
                pool.submit(seg_stat, bs_size, ci_confidence, in_dir, out_dir, model_key)
            )

    for future in futures:
        future.result()


if __name__ == "__main__":
    main()
