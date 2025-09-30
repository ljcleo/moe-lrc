from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig

from misc import get_global_config, get_model_keys, model_config, print_log


def tok_stat(min_freq, top_k, model_dir, tok_dir, pred_dir, rank_dir, out_dir, model_key):
    print_log("working on", model_key, "...")
    all_tokens = load_file(tok_dir / f"{model_key}.safetensors")
    all_preds = load_file(pred_dir / f"{model_key}.safetensors")

    vocab_size = AutoConfig.from_pretrained(
        model_dir / model_key, trust_remote_code=True
    ).vocab_size

    def make_map(tokens: torch.Tensor):
        return (tokens.ravel().bincount(minlength=vocab_size) >= min_freq).nonzero().ravel()

    all_token_map = {data_key: make_map(tokens) for data_key, tokens in all_tokens.items()}
    all_pred_map = {data_key: make_map(preds) for data_key, preds in all_preds.items()}

    all_gold_map = {data_key: make_map(tokens[:, 1:]) for data_key, tokens in all_tokens.items()}

    results = {
        "cnt": {},
        "map": {
            **{f"{data_key}_in": token_map for data_key, token_map in all_token_map.items()},
            **{f"{data_key}_pred": pred_map for data_key, pred_map in all_pred_map.items()},
            **{f"{data_key}_out": gold_map for data_key, gold_map in all_gold_map.items()},
        },
    }

    for layer_idx in range(model_config.loc[model_key, "num_layers"]):
        stem = f"{model_key}_{layer_idx:02d}"
        file = rank_dir / f"{stem}.safetensors"

        if file.exists():
            print_log("working on", stem, "...")
            num_experts = model_config.loc[model_key, "num_experts"]

            for data_key, rank in load_file(file).items():
                mask = rank >= num_experts - top_k
                key = f"{layer_idx:02d}_{data_key}"

                def make_stat(tokens: torch.Tensor, mask: torch.Tensor, token_map: torch.Tensor):
                    return torch.stack(
                        [
                            tokens[mask[..., i]].bincount(minlength=vocab_size)[token_map]
                            for i in range(num_experts)
                        ]
                    )

                results["cnt"][f"{key}_in"] = make_stat(
                    all_tokens[data_key], mask, all_token_map[data_key]
                )

                results["cnt"][f"{key}_pred"] = make_stat(
                    all_preds[data_key], mask, all_pred_map[data_key]
                )

                results["cnt"][f"{key}_out"] = make_stat(
                    all_tokens[data_key][:, 1:], mask[..., :-1, :], all_gold_map[data_key]
                )

            print_log(stem, "finished.")

    if len(results["cnt"]) > 0:
        for k, v in results.items():
            print_log("saving", model_key, k, "...")
            save_file(v, out_dir / f"{model_key}_{k}.safetensors")

    print_log(model_key, "finished.")


def main():
    model_keys = get_model_keys(from_arg=True)
    min_freq: int = get_global_config("vocab_min_freq")
    top_k: int = get_global_config("vocab_top_k")

    data_dir = Path("../data")
    model_dir = Path("../model")
    output_dir = Path("../output")
    tok_dir = data_dir / "merged"
    pred_dir = output_dir / "pred"
    rank_dir = output_dir / "rank"
    out_dir = output_dir / "vocab"
    out_dir.mkdir(parents=True, exist_ok=True)

    futures = []

    with ProcessPoolExecutor(max_workers=2) as pool:
        for model_key in model_keys:
            if model_config.loc[model_key, "type"] == "causal":
                futures.append(
                    pool.submit(
                        tok_stat,
                        min_freq,
                        top_k,
                        model_dir,
                        tok_dir,
                        pred_dir,
                        rank_dir,
                        out_dir,
                        model_key,
                    )
                )

    for future in futures:
        future.result()


if __name__ == "__main__":
    main()
