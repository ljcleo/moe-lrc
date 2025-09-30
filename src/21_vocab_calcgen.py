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
    num_experts = model_config.loc[model_key, "num_experts"]

    vocab_size = AutoConfig.from_pretrained(
        model_dir / model_key, trust_remote_code=True
    ).vocab_size

    token_map = (
        (
            torch.stack(
                [tokens.ravel().bincount(minlength=vocab_size) for tokens in all_tokens.values()]
            ).sum(dim=0)
            >= min_freq
        )
        .nonzero()
        .ravel()
    )

    pred_map = (
        (
            torch.stack(
                [preds.ravel().bincount(minlength=vocab_size) for preds in all_preds.values()]
            ).sum(dim=0)
            >= min_freq
        )
        .nonzero()
        .ravel()
    )

    gold_map = (
        (
            torch.stack(
                [
                    tokens[:, 1:].ravel().bincount(minlength=vocab_size)
                    for tokens in all_tokens.values()
                ]
            ).sum(dim=0)
            >= min_freq
        )
        .nonzero()
        .ravel()
    )

    results = {"cnt": {}, "map": {"in": token_map, "pred": pred_map, "out": gold_map}}

    def add_assign(key, value):
        if key not in results["cnt"]:
            results["cnt"][key] = value
        else:
            results["cnt"][key] = results["cnt"][key] + value

    for layer_idx in range(model_config.loc[model_key, "num_layers"]):
        stem = f"{model_key}_{layer_idx:02d}"
        file = rank_dir / f"{stem}.safetensors"

        if file.exists():
            print_log("working on", stem, "...")

            for data_key, rank in load_file(file).items():
                mask = rank >= num_experts - top_k
                key = f"{layer_idx:02d}"

                def make_stat(tokens: torch.Tensor, mask: torch.Tensor, token_map: torch.Tensor):
                    return torch.stack(
                        [
                            tokens[mask[..., i]].bincount(minlength=vocab_size)[token_map]
                            for i in range(num_experts)
                        ]
                    )

                add_assign(f"{key}_in", make_stat(all_tokens[data_key], mask, token_map))
                add_assign(f"{key}_pred", make_stat(all_preds[data_key], mask, pred_map))

                add_assign(
                    f"{key}_out",
                    make_stat(all_tokens[data_key][:, 1:], mask[..., :-1, :], gold_map),
                )

            print_log(stem, "finished.")

    if len(results["cnt"]) > 0:
        for k, v in results.items():
            print_log("saving", model_key, k, "...")
            save_file(v, out_dir / f"{model_key}_{k}_gen.safetensors")

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
