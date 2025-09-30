from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
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
    top_k = model_config.loc[model_key, "top_k"]
    stats = {key: [] for key in ("m", "l")}

    for df_name, seg_stat in load_file(file).items():
        layer_idx, data_key, seg_len = df_name.split("_")
        seg_len = int(seg_len)
        base_size = seg_stat.shape[1] * top_k // num_experts

        if layer_idx in "de":
            is_decoder = layer_idx == "d"

            stats["m"].extend(
                (
                    {
                        "model": model_key,
                        "is_decoder": is_decoder,
                        "dataset": data_key,
                        "seg_len": seg_len,
                        "cache_m": (idx + 1) / base_size,
                        "recall": recall[1].item(),
                        "ci_lb": recall[0].item(),
                        "ci_ub": recall[2].item(),
                    }
                    for idx, recall in enumerate(seg_stat.mT)
                )
            )
        else:
            layer_idx = int(layer_idx)

            if is_seq2seq:
                if layer_idx >= num_layers // 2:
                    is_decoder = True
                    layer_idx -= num_layers // 2
                else:
                    is_decoder = False
            else:
                is_decoder = True

            stats["l"].extend(
                (
                    {
                        "model": model_key,
                        "is_decoder": is_decoder,
                        "layer_idx": layer_idx,
                        "dataset": data_key,
                        "seg_len": seg_len,
                        "cache_m": (idx + 1) / base_size,
                        "recall": recall[1].item(),
                        "ci_lb": recall[0].item(),
                        "ci_ub": recall[2].item(),
                    }
                    for idx, recall in enumerate(seg_stat.mT)
                )
            )

    for df_name, stat in stats.items():
        df_dtypes = {"seg_len": np.uint16}
        if df_name == "l":
            df_dtypes["layer_idx"] = np.uint8

        pd.DataFrame(stat).astype(df_dtypes).to_parquet(out_dir / f"{model_key}_{df_name}.parquet")

    print_log(model_key, "finished.")


def main():
    model_keys = get_model_keys(from_arg=True)

    root_dir = Path("../output")
    in_dir = root_dir / "sch"
    out_dir = root_dir / "sch_pq"
    out_dir.mkdir(parents=True, exist_ok=True)

    futures = []

    with ProcessPoolExecutor(max_workers=20) as pool:
        for model_key in model_keys:
            futures.append(pool.submit(refine, in_dir, out_dir, model_key))

    for future in futures:
        future.result()


if __name__ == "__main__":
    main()
