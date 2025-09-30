from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from safetensors.torch import load_file

from misc import get_data_keys, get_model_keys, print_log


def refine(loss_dir, model_key: str) -> list[dict[str, str | int | float]]:
    file = loss_dir / f"{model_key}.safetensors"
    results = []

    if file.exists():
        print_log("working on", model_key, "...")

        for data_key, losses in load_file(file).items():
            results.append(
                {
                    "model": model_key,
                    "dataset": data_key,
                    "loss": losses.mean().item(),
                }
            )

        print_log(model_key, "finished.")

    return results


def main():
    root_dir = Path("../output")
    loss_dir = root_dir / "loss"
    out_file = root_dir / "loss.parquet"

    model_keys = get_model_keys()
    with ProcessPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(refine, loss_dir, model_key) for model_key in model_keys]

    all_results: list[dict[str, str | int | float]] = []
    for future in futures:
        all_results.extend(future.result())

    df = pd.DataFrame(all_results)
    df["model"] = df["model"].astype(pd.CategoricalDtype(model_keys, ordered=True))
    df["dataset"] = df["dataset"].astype(pd.CategoricalDtype(get_data_keys(), ordered=True))
    df.sort_values(df.columns.tolist()).to_parquet(out_file)


if __name__ == "__main__":
    main()
