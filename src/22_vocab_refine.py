from concurrent.futures import Future, ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors.torch import load_file
from transformers import AutoTokenizer

from misc import get_data_keys, get_global_config, get_model_keys, print_log


def calc_hifreq(df: pd.DataFrame, *, count: int):
    freq = df["freq"].mean()
    df = df.sort_values(["freq", "count"], ascending=False).head(count)
    return pd.Series({"freq": freq, "hitoken": df["token"].values, "hifreq": df["freq"].values})


def refine(hifreq_count: int, in_dir: Path, model_key: str, is_gen: bool) -> pd.DataFrame | None:
    stat_type = "gen" if is_gen else "dat"
    suffix = "_gen" if is_gen else ""
    cnt_file = in_dir / f"{model_key}_cnt{suffix}.safetensors"
    map_file = in_dir / f"{model_key}_map{suffix}.safetensors"

    if not (cnt_file.exists() and map_file.exists()):
        return None

    dfs = []
    print_log("working on", model_key, stat_type, "...")
    tokenizer = AutoTokenizer.from_pretrained(f"./model/{model_key}", trust_remote_code=True)

    vocab = {
        map_key: np.array(
            [
                tokenizer.decode(
                    [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                for token_id in token_ids
            ]
        )
        for map_key, token_ids in load_file(map_file).items()
    }

    main_cols = (
        ["model", "layer_idx", "expert_idx"] + ([] if is_gen else ["dataset"]) + ["token_type"]
    )

    for key, tok_stat in load_file(cnt_file).items():
        if is_gen:
            layer_idx, token_type = key.split("_")
            map_key = token_type
        else:
            layer_idx, data_key, token_type = key.split("_")
            map_key = f"{data_key}_{token_type}"

        layer_idx = int(layer_idx)
        tok_count = tok_stat.sum(dim=0)
        tok_expert_freq, tok_expert_idx = (tok_stat / tok_count).max(dim=0)

        ldf = pd.DataFrame(
            {
                "token": vocab[map_key],
                "count": tok_count,
                "expert_idx": tok_expert_idx,
                "freq": tok_expert_freq,
            }
        ).assign(model=model_key, layer_idx=layer_idx, token_type=token_type)

        if not is_gen:
            ldf["dataset"] = data_key

        dfs.append(ldf[main_cols + ["token", "count", "freq"]])

    df = (
        pd.concat(dfs)
        .groupby(main_cols, as_index=False)
        .apply(partial(calc_hifreq, count=hifreq_count), include_groups=False)
    )

    print_log(model_key, stat_type, "finished.")
    return df


def main():
    hifreq_count = get_global_config("vocab_hifreq_count")

    root_dir = Path("../output")
    in_dir = root_dir / "vocab"
    out_dir = root_dir / "vocab_pq"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_futures: dict[str, list[Future[pd.DataFrame | None]]] = {k: [] for k in ("gen",)}  # "dat")}
    model_keys = get_model_keys()

    with ProcessPoolExecutor(max_workers=16) as pool:
        for model_key in model_keys:
            for stat_type, futures in all_futures.items():
                futures.append(
                    pool.submit(refine, hifreq_count, in_dir, model_key, stat_type == "gen")
                )

    all_dfs: dict[str, list[pd.DataFrame]] = {k: [] for k in all_futures.keys()}

    for stat_type, futures in all_futures.items():
        for future in futures:
            result = future.result()
            if result is not None:
                all_dfs[stat_type].append(result)

    model_dtype = pd.CategoricalDtype(model_keys, ordered=True)
    data_dtype = pd.CategoricalDtype(get_data_keys(), ordered=True)

    for stat_type, dfs in all_dfs.items():
        df = pd.concat(dfs, ignore_index=True)
        df["model"] = df["model"].astype(model_dtype)

        if "dataset" in df.columns:
            df["dataset"] = df["dataset"].astype(data_dtype)

        df.sort_values(df.columns.tolist()[:-2]).to_parquet(out_dir / f"{stat_type}.parquet")


if __name__ == "__main__":
    main()
