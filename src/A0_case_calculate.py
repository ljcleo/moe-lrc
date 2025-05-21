import os
from pathlib import Path

import orjson
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from misc import get_global_config, get_model_key, model_config, print_log


def main():
    model_key = get_model_key()
    assert model_config.loc[model_key, "type"] == "causal"
    gen_length: int = get_global_config("case_gen_length")

    model_dir = Path("../model")
    case_dir = Path("../case")
    in_dir = case_dir / "json"
    token_dir = case_dir / "tokenized"
    log_dir = case_dir / "raw"
    gen_dir = case_dir / "gen"
    token_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)

    os.environ["LRC_LOG_PATH"] = log_dir.as_posix()
    os.environ["LRC_MODEL_KEY"] = model_key
    os.environ["LRC_DUMP_SIZE"] = "1"

    model_path = model_dir / model_key
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    device_map_path = model_path / "device_map.json"
    device_map = orjson.loads(device_map_path.read_bytes()) if device_map_path.exists() else "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        attn_implementation=model_config.loc[model_key, "attn"],
    )

    model.generation_config = GenerationConfig.from_pretrained(model_path)

    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        if isinstance(model.generation_config.pad_token_id, list):
            model.generation_config.pad_token_id = model.generation_config.pad_token_id[0]

    with (in_dir / "cases.jsonl").open(encoding="utf8") as f:
        prompts = [orjson.loads(line) for line in f]

    all_tokens = []
    all_responses = []

    for num_dumps, prompt in enumerate(prompts):
        data_key = prompt["dataset"]
        print_log("working on", data_key, "...")
        input = tokenizer(prompt["prompt"], return_tensors="pt", return_token_type_ids=False)

        try:
            tokens = input.tokens()
        except ValueError:
            tokens = tokenizer.convert_ids_to_tokens(input.input_ids[0])

        all_tokens.append({"dataset": data_key, "tokens": tokens})

        outputs = model.generate(
            **input.to(model.device),
            max_new_tokens=gen_length,
            num_beams=1,
            do_sample=False,
            temperature=None,
            top_p=1,
        )

        response = tokenizer.decode(outputs[0][len(tokens) :], skip_special_tokens=True)
        all_responses.append({"dataset": data_key, "response": response})
        print_log(f"{data_key} generated:", response)

        for layer_idx in range(model_config.loc[model_key, "num_layers"]):
            stem = f"{model_key}_{layer_idx:02d}"
            old_file = log_dir / f"{stem}.{num_dumps}.pt"

            if old_file.exists():
                old_file.rename(log_dir / f"{stem}_{data_key}.pt")

    with (token_dir / f"{model_key}.jsonl").open("wb") as f:
        for tokens in all_tokens:
            f.write(orjson.dumps(tokens, option=orjson.OPT_APPEND_NEWLINE))

    with (gen_dir / f"{model_key}.jsonl").open("wb") as f:
        for response in all_responses:
            f.write(orjson.dumps(response, option=orjson.OPT_APPEND_NEWLINE))


if __name__ == "__main__":
    main()
