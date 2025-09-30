import os
from functools import partial
from pathlib import Path
from typing import Any, Callable

import orjson
import torch
from safetensors.torch import load_file, save_file
from torch.nn.functional import cross_entropy
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.trainer_utils import enable_full_determinism

from misc import get_global_config, get_model_key, model_config, print_log


def calc_switch(
    model, input_ids: torch.Tensor, *, extra_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = input_ids.cuda()

    mask_count = get_global_config("switch_mask_count")
    step = input_ids.shape[0] // mask_count
    mask_pos = torch.arange(mask_count).cuda() * step + step // 2
    mask_ids = extra_ids[:mask_count].cuda()

    labels = torch.cat(
        [
            torch.stack([mask_ids, input_ids[mask_pos]], dim=1).ravel(),
            extra_ids[mask_count : mask_count + 1].cuda(),
        ]
    )

    input_ids[mask_pos] = mask_ids

    decoder_input_ids = torch.cat(
        [torch.tensor([model.config.decoder_start_token_id]).cuda(), labels[:-1]]
    )

    logits: torch.Tensor = model(
        input_ids=input_ids.unsqueeze(0),
        decoder_input_ids=decoder_input_ids.unsqueeze(0),
        use_cache=False,
    ).logits[0]

    pred = logits.argmax(dim=1)
    loss = cross_entropy(logits.float(), labels)
    return pred, loss


def calc_nllb(model, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = input_ids.cuda()

    decoder_input_ids = torch.cat(
        [torch.tensor([model.config.decoder_start_token_id]).cuda(), input_ids[:-1]]
    )

    logits: torch.Tensor = model(
        input_ids=input_ids.unsqueeze(0),
        decoder_input_ids=decoder_input_ids.unsqueeze(0),
        use_cache=False,
    ).logits[0]

    pred = logits.argmax(dim=1)
    loss = cross_entropy(logits.float(), input_ids)
    return pred, loss


def calc_other(model, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = input_ids.cuda()
    logits: torch.Tensor = model(input_ids=input_ids.unsqueeze(0), use_cache=False).logits[0]
    pred = logits.argmax(dim=1)
    loss = cross_entropy(logits.float()[:-1], input_ids[1:])
    return pred, loss


def main():
    enable_full_determinism(19260817)
    model_key = get_model_key()

    model_dir = Path("../model")
    in_dir = Path("../data/merged")
    out_dir = Path("../output")
    log_dir = out_dir / "raw"
    pred_dir = out_dir / "pred"
    loss_dir = out_dir / "loss"
    log_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    loss_dir.mkdir(parents=True, exist_ok=True)

    os.environ["LRC_LOG_PATH"] = log_dir.as_posix()
    os.environ["LRC_MODEL_KEY"] = model_key
    os.environ["LRC_DUMP_SIZE"] = str(get_global_config("num_samples"))

    model_path = model_dir / model_key
    calc_switch_local = calc_switch

    if model_key == "switch":
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        extra_ids = torch.tensor(
            tokenizer.convert_tokens_to_ids(
                tokenizer.special_tokens_map["additional_special_tokens"]
            )
        )

        calc_switch_local = partial(calc_switch, extra_ids=extra_ids)

    auto_model_class: dict[str, type[AutoModelForCausalLM] | type[AutoModelForSeq2SeqLM]] = {
        "causal": AutoModelForCausalLM,
        "seq2seq": AutoModelForSeq2SeqLM,
    }

    calc_func: dict[str, Callable[[Any, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]] = {
        "switch": calc_switch_local,
        "nllb": calc_nllb,
    }

    device_map_path = model_path / "device_map.json"
    device_map = orjson.loads(device_map_path.read_bytes()) if device_map_path.exists() else "auto"

    model = (
        auto_model_class[model_config.loc[model_key, "type"]]
        .from_pretrained(
            model_dir / model_key,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            attn_implementation=model_config.loc[model_key, "attn"],
        )
        .eval()
    )

    all_preds = {}
    all_losses = {}

    with torch.no_grad():
        for num_dumps, (data_key, input_ids) in enumerate(
            load_file(in_dir / f"{model_key}.safetensors").items()
        ):
            print_log("working on", data_key, "...")
            preds = []
            losses = []

            for cur_input_ids in tqdm(input_ids):
                pred, loss = calc_func.get(model_key, calc_other)(model, cur_input_ids)
                preds.append(pred.cpu())
                losses.append(loss.cpu())

            all_preds[data_key] = torch.stack(preds)
            all_losses[data_key] = torch.stack(losses)
            print_log(f"{data_key} loss", all_losses[data_key].mean().item())

            for layer_idx in range(model_config.loc[model_key, "num_layers"]):
                stem = f"{model_key}_{layer_idx:02d}"
                old_file = log_dir / f"{stem}.{num_dumps}.pt"

                if old_file.exists():
                    old_file.rename(log_dir / f"{stem}_{data_key}.pt")

    save_file(all_preds, pred_dir / f"{model_key}.safetensors")
    save_file(all_losses, loss_dir / f"{model_key}.safetensors")


if __name__ == "__main__":
    main()
