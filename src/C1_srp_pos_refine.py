from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

from misc import get_model_keys, model_config, print_log


def refine(in_dir, out_dir, model_key: str) -> None:
    file = in_dir / f"{model_key}.safetensors"
    if not file.exists():
        return None

    print_log("working on", model_key, "...")
    is_seq2seq = model_config.loc[model_key, "type"] == "seq2seq"
    num_layers = model_config.loc[model_key, "num_layers"]
    num_experts = model_config.loc[model_key, "num_experts"]

    all_stats: dict[str, dict[tuple[str, ...], dict[str, torch.Tensor]]] = {
        f"{stat_level}{stat_type}": {} for stat_level in "mle" for stat_type in "gd"
    }

    def assign_add(df_name: str, key: tuple[str, ...], **kwargs: torch.Tensor) -> None:
        cur_stats = all_stats[df_name]
        if key not in cur_stats:
            cur_stats[key] = kwargs
        else:
            for k, v in kwargs.items():
                cur_stats[key][k] = cur_stats[key][k] + v

    for key, seg_stat in load_file(file).items():
        print_log("merging", model_key, "tensor key", key, "...")
        layer_idx, data_key, seg_len = key.split("_")
        layer_idx = int(layer_idx)
        seg_len = int(seg_len)

        if is_seq2seq:
            if layer_idx >= num_layers // 2:
                is_decoder = True
                layer_idx -= num_layers // 2
            else:
                is_decoder = False
        else:
            is_decoder = True

        assert seg_stat.shape[0] == num_experts
        assert seg_stat.shape[-1] == seg_len + 1
        num_segs = seg_stat.sum(dim=-1, keepdim=True)
        num_cases = num_segs * seg_len

        num_matches = (
            (seg_stat * torch.arange(seg_len + 1)).flip(dims=(-1,)).cumsum(dim=-1).flip(dims=(-1,))
        )

        num_preds = seg_stat.flip(dims=(-1,)).cumsum(dim=-1).flip(dims=(-1,)) * seg_len
        num_acts = num_matches[..., :1]

        cur_stats = {
            "num_cases": num_cases,
            "num_matches": num_matches,
            "num_preds": num_preds,
            "num_acts": num_acts,
        }

        key_prefix = (is_decoder,)
        key_suffix = (seg_len,)
        assign_add("eg", key_prefix + (layer_idx,) + key_suffix, **cur_stats)
        assign_add("ed", key_prefix + (layer_idx, data_key) + key_suffix, **cur_stats)

        sum_stats = {k: v.sum(dim=0) for k, v in cur_stats.items()}
        assign_add("mg", key_prefix + key_suffix, **sum_stats)
        assign_add("lg", key_prefix + (layer_idx,) + key_suffix, **sum_stats)
        assign_add("md", key_prefix + (data_key,) + key_suffix, **sum_stats)
        assign_add("ld", key_prefix + (layer_idx, data_key) + key_suffix, **sum_stats)

    for df_name, stats in all_stats.items():
        print_log("generating", model_key, "dataframe", df_name, "...")
        stat_level = df_name[0]
        stat_type = df_name[1]
        recs = []

        for (is_decoder, *sub_keys, seg_len), sub_stats in stats.items():
            act_r = (sub_stats["num_acts"] / sub_stats["num_cases"]).squeeze(-1)

            f1 = (
                sub_stats["num_matches"]
                * 2
                / (sub_stats["num_preds"] + sub_stats["num_acts"]).clamp(min=1)
            )

            best_f1, best_p = f1.max(dim=-1)

            best_m = (
                sub_stats["num_preds"].take_along_dim(best_p.unsqueeze(-1), dim=-1)
                * (sub_stats["num_acts"] > 0)
                / sub_stats["num_acts"].clamp(min=1)
            ).squeeze(-1)

            srp = {"act_r": act_r, "best_f1": best_f1, "best_m": best_m}

            common = {
                "model": model_key,
                "is_decoder": is_decoder,
                "seg_len": seg_len,
                "start_pos": torch.arange(srp["act_r"].shape[-1]).numpy().astype(np.uint16),
            }

            if stat_level == "e":
                srp_list = [
                    {"expert_idx": expert_idx}
                    | common
                    | {k: v[expert_idx].numpy() for k, v in srp.items()}
                    for expert_idx in range(num_experts)
                ]
            else:
                srp_dict = common | {k: v.numpy() for k, v in srp.items()}

            extra = {}
            if stat_type == "d":
                *sub_keys, extra["dataset"] = sub_keys

            if stat_level == "m":
                () = sub_keys
            else:
                (extra["layer_idx"],) = sub_keys

            if stat_level == "e":
                recs.extend((extra | srp_sub for srp_sub in srp_list))
            else:
                recs.append(extra | srp_dict)

        key_prefix = ["model", "is_decoder"]
        key_suffix = ["seg_len", "start_pos", "act_r", "best_f1", "best_m"]
        key_infix = []
        int_dtypes = {"seg_len": np.uint16}

        if stat_level != "m":
            key_infix.append("layer_idx")
            int_dtypes["layer_idx"] = np.uint8
        if stat_level == "e":
            key_infix.append("expert_idx")
            int_dtypes["expert_idx"] = np.uint8
        if stat_type == "d":
            key_infix.append("dataset")

        pd.DataFrame(recs).astype(int_dtypes)[key_prefix + key_infix + key_suffix].explode(
            key_suffix[1:], ignore_index=True
        ).astype({"start_pos": np.uint16}).to_parquet(out_dir / f"{model_key}_{df_name}.parquet")

    print_log(model_key, "finished.")


def main():
    model_keys = get_model_keys(from_arg=True)

    root_dir = Path("../output")
    in_dir = root_dir / "srp_pos"
    out_dir = root_dir / "srp_pos_pq"
    out_dir.mkdir(parents=True, exist_ok=True)

    futures = []

    with ProcessPoolExecutor(max_workers=2) as pool:
        for model_key in model_keys:
            futures.append(pool.submit(refine, in_dir, out_dir, model_key))

    for future in futures:
        future.result()


if __name__ == "__main__":
    main()
