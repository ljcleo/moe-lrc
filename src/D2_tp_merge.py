from pathlib import Path

import pandas as pd
from misc import get_data_keys, get_model_keys, print_log


def main():
    root_dir = Path("../output")
    in_dir = root_dir / "tp_pq"
    out_dir = root_dir / "tp_mpq"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dtype = pd.CategoricalDtype(get_model_keys(), ordered=True)
    data_dtype = pd.CategoricalDtype(get_data_keys(), ordered=True)
    mode_dtype = pd.CategoricalDtype(["old_doc", "new_gen"], ordered=True)

    print_log("working on m ...")
    df: pd.DataFrame = pd.concat((pd.read_parquet(f) for f in in_dir.iterdir()), ignore_index=True)

    df["model"] = df["model"].astype(model_dtype)
    df["dataset"] = df["dataset"].astype(data_dtype)
    df["mode"] = df["mode"].astype(mode_dtype)

    df["cache_m"] = df["cache_m"].astype(float)
    df.loc[df["cache_m"] > 3, "cache_m"] = float("inf")

    print_log("saving m ...")
    df.sort_values(df.columns.tolist(), ignore_index=True).to_parquet(out_dir / "m.parquet")
    print_log("m saved.")


if __name__ == "__main__":
    main()
