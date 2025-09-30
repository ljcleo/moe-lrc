from pathlib import Path

import pandas as pd

from misc import get_data_keys, get_model_keys, print_log


def main():
    root_dir = Path("../output")
    in_dir = root_dir / "srp_pos_pq"
    out_dir = root_dir / "srp_pos_mpq"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dtype = pd.CategoricalDtype(get_model_keys(), ordered=True)
    data_dtype = pd.CategoricalDtype(get_data_keys(), ordered=True)

    for df_name in ("mg", "lg", "eg", "md", "ld", "ed"):
        print_log("working on", df_name, "...")
        df = pd.concat([pd.read_parquet(file) for file in in_dir.glob(f"*_{df_name}.parquet")])

        df["model"] = df["model"].astype(model_dtype)
        if "dataset" in df.columns:
            df["dataset"] = df["dataset"].astype(data_dtype)

        print_log("saving", df_name, "...")

        df.sort_values(df.columns.tolist(), ignore_index=True).to_parquet(
            out_dir / f"{df_name}.parquet"
        )

        print_log(df_name, "saved.")


if __name__ == "__main__":
    main()
