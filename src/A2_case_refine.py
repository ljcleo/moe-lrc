from pathlib import Path

import pandas as pd
import torch
from safetensors.torch import load_file

from misc import get_data_keys, get_global_config, get_model_keys, model_config


def main():
    model_keys = get_model_keys()
    seg_lens = get_global_config("seg_lengths")

    root_dir = Path("../case")
    token_dir = root_dir / "tokenized"
    log_dir = root_dir / "merged"

    results = {stat_type: [] for stat_type in ("tokens", "logits", "srp")}

    for model_key in model_keys:
        token_file = token_dir / f"{model_key}.jsonl"
        log_file = log_dir / f"{model_key}.safetensors"

        if token_file.exists() and log_file.exists():
            tokens = pd.read_json(token_file, lines=True).set_index("dataset")["tokens"]
            tokens_added = set()

            num_experts = model_config.loc[model_key, "num_experts"]
            top_k = model_config.loc[model_key, "top_k"]

            for key, logits in load_file(log_file).items():
                layer_idx, data_key = key.split("_")
                common = {"model": model_key, "dataset": data_key}

                if data_key not in tokens_added:
                    results["tokens"].extend(
                        (
                            {**common, "pos": pos, "token": token}
                            for pos, token in enumerate(tokens[data_key][: seg_lens[-1]])
                        )
                    )

                    tokens_added.add(data_key)

                layer_idx = int(layer_idx)
                common["layer_idx"] = layer_idx
                logits = logits[: seg_lens[-1]]

                results["logits"].extend(
                    (
                        {**common, "expert_idx": expert_idx, "pos": pos, "logit": logit.item()}
                        for expert_idx in range(num_experts)
                        for pos, logit in enumerate(logits[:, expert_idx].float())
                    )
                )

                rank = logits.argsort(dim=-1).argsort(dim=-1).to(torch.int8)
                mask = (rank >= num_experts - top_k).to(torch.int16)
                seg_len = 1

                while seg_len < seg_lens[-1]:
                    mask = mask[:-seg_len] + mask[seg_len:]
                    seg_len <<= 1

                    if seg_len in seg_lens:
                        seg_stat = torch.stack(
                            [
                                mask[:, expert_idx].bincount(minlength=seg_len + 1)
                                for expert_idx in range(num_experts)
                            ]
                        )

                        num_segs = seg_stat.sum(dim=-1, keepdim=True)
                        num_cases = num_segs * seg_len

                        num_matches = (
                            (seg_stat * torch.arange(seg_len + 1)).fliplr().cumsum(dim=1).fliplr()
                        )

                        num_preds = seg_stat.fliplr().cumsum(dim=1).fliplr() * seg_len
                        num_acts = num_matches[:, :1]

                        act_r = (num_acts / num_cases).squeeze(-1)
                        f1 = num_matches * 2 / (num_preds + num_acts).clamp(min=1)
                        best_f1, best_p = f1.max(dim=-1)

                        best_m = (
                            num_preds.take_along_dim(best_p.unsqueeze(-1), dim=-1)
                            * (num_acts > 0)
                            / num_acts.clamp(min=1)
                        ).squeeze(-1)

                        results["srp"].extend(
                            (
                                {
                                    **common,
                                    "expert_idx": expert_idx,
                                    "seg_len": seg_len,
                                    "act_r": act_r[expert_idx].item(),
                                    "best_f1": best_f1[expert_idx].item(),
                                    "best_m": best_m[expert_idx].item(),
                                }
                                for expert_idx in range(num_experts)
                            )
                        )

    model_dtype = pd.CategoricalDtype(model_keys, ordered=True)
    data_dtype = pd.CategoricalDtype(get_data_keys(), ordered=True)

    for df_name, result in results.items():
        df = pd.DataFrame(result)

        df["model"] = df["model"].astype(model_dtype)
        if "dataset" in df.columns:
            df["dataset"] = df["dataset"].astype(data_dtype)

        df.to_parquet(root_dir / f"{df_name}.parquet")


if __name__ == "__main__":
    main()
