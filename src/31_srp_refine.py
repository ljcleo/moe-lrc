from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

from misc import get_global_config, get_model_keys, model_config, print_log


def refine(ci_confidence, in_dir, out_dir, model_key: str) -> None:
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

    def assign_add(df_name: str, key: tuple[str, ...], sub_key: str, value: torch.Tensor) -> None:
        cur_stats = all_stats[df_name]

        if key not in cur_stats:
            cur_stats[key] = {sub_key: value}
        else:
            sub_stats = cur_stats[key]

            if sub_key not in sub_stats:
                sub_stats[sub_key] = value
            else:
                sub_stats[sub_key] = sub_stats[sub_key] + value

    for key, value in load_file(file).items():
        print_log(key)
        layer_idx, data_key, seg_len, stat_type = key.split("_")
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

        key_prefix = (is_decoder,)
        key_suffix = (seg_len,)
        assign_add("eg", key_prefix + (layer_idx,) + key_suffix, stat_type, value)
        assign_add("ed", key_prefix + (layer_idx, data_key) + key_suffix, stat_type, value)

        value = value.sum(dim=1)
        assign_add("mg", key_prefix + key_suffix, stat_type, value)
        assign_add("lg", key_prefix + (layer_idx,) + key_suffix, stat_type, value)
        assign_add("md", key_prefix + (data_key,) + key_suffix, stat_type, value)
        assign_add("ld", key_prefix + (layer_idx, data_key) + key_suffix, stat_type, value)

    for df_name, stats in all_stats.items():
        print_log(df_name)
        stat_level = df_name[0]
        stat_type = df_name[1]
        recs = []

        for (is_decoder, *sub_keys, seg_len), sub_stats in stats.items():
            act_r = (sub_stats["a"][0] / sub_stats["p"][0, ..., :1]).squeeze(-1)
            f1 = sub_stats["m"] * 2 / (sub_stats["p"] + sub_stats["a"]).clamp(min=1)
            best_f1, best_p = f1[0].max(dim=-1)

            best_f1_bs = f1[1:].max(dim=-1).values.sort(dim=0).values
            bs_size = best_f1_bs.shape[0]
            bs_lp = round(bs_size * ((1 - ci_confidence) / 2))
            bs_up = bs_size - bs_lp
            best_f1_l, best_f1_u = best_f1 * 2 - best_f1_bs[[bs_up, bs_lp]]

            best_m = (
                sub_stats["p"][0].take_along_dim(best_p.unsqueeze(-1), dim=-1)
                * (sub_stats["a"][0] > 0)
                / sub_stats["a"][0].clamp(min=1)
            ).squeeze(-1)

            srp = {
                "act_r": act_r,
                "best_f1": best_f1,
                "ci_lb": best_f1_l,
                "ci_ub": best_f1_u,
                "best_m": best_m,
            }

            common = {"model": model_key, "is_decoder": is_decoder, "seg_len": seg_len}

            if stat_level == "e":
                srp_list = [
                    {"expert_idx": expert_idx}
                    | common
                    | {k: v[expert_idx].item() for k, v in srp.items()}
                    for expert_idx in range(num_experts)
                ]
            else:
                srp_dict = common | {k: v.item() for k, v in srp.items()}

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
        key_suffix = ["seg_len", "act_r", "best_f1", "ci_lb", "ci_ub", "best_m"]
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

        pd.DataFrame(recs).astype(int_dtypes)[key_prefix + key_infix + key_suffix].to_parquet(
            out_dir / f"{model_key}_{df_name}.parquet"
        )

    print_log(model_key, "finished.")


def main():
    model_keys = get_model_keys(from_arg=True)
    ci_confidence: float = get_global_config("ci_confidence")

    root_dir = Path("../output")
    in_dir = root_dir / "srp"
    out_dir = root_dir / "srp_pq"
    out_dir.mkdir(parents=True, exist_ok=True)

    futures = []

    with ProcessPoolExecutor(max_workers=2) as pool:
        for model_key in model_keys:
            futures.append(pool.submit(refine, ci_confidence, in_dir, out_dir, model_key))

    for future in futures:
        future.result()


if __name__ == "__main__":
    main()
