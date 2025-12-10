from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from misc import get_model_keys, print_log
from safetensors.torch import load_file


def refine(in_dir: Path, out_dir: Path, model_key: str) -> None:
    files = sorted(in_dir.glob(f"{model_key}_*.safetensors"))
    if len(files) == 0:
        return None

    print_log("working on", model_key, "...")
    stats = []

    for file in files:
        cache_m = int(file.stem.partition("_")[-1])

        for df_name, records in load_file(file).items():
            data_key, _, mode = df_name.partition("_")
            prefill_time, decode_time = records.median(dim=0).values.tolist()
            prefill_thrp = 480 / prefill_time
            decode_thrp = 32 / decode_time

            stats.append(
                {
                    "model": model_key,
                    "dataset": data_key,
                    "cache_m": cache_m,
                    "mode": mode,
                    "prefill_thrp": prefill_thrp,
                    "decode_thrp": decode_thrp,
                }
            )

    pd.DataFrame(stats).to_parquet(out_dir / f"{model_key}.parquet")
    print_log(model_key, "finished.")


def main():
    model_keys = get_model_keys(from_arg=True)

    root_dir = Path("../output")
    in_dir = root_dir / "tp"
    out_dir = root_dir / "tp_pq"
    out_dir.mkdir(parents=True, exist_ok=True)

    futures = []

    with ProcessPoolExecutor(max_workers=20) as pool:
        for model_key in model_keys:
            futures.append(pool.submit(refine, in_dir, out_dir, model_key))

    for future in futures:
        future.result()


if __name__ == "__main__":
    main()
