import os
from pathlib import Path
from time import time

import torch
from misc import get_model_key, model_config, print_log
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers.trainer_utils import enable_full_determinism


def measure_time(model, input_ids: torch.Tensor, gen_new) -> tuple[float, float]:
    input_ids = input_ids.cuda(0)
    prompt, as_gen = input_ids.split([480, 32])

    torch.cuda.synchronize()
    start_time = time()
    output = model(input_ids=prompt.unsqueeze(0), use_cache=True)
    torch.cuda.synchronize()
    prefill_time = time() - start_time

    torch.cuda.synchronize()
    start_time = time()

    for i in range(as_gen.shape[0]):
        output = model(
            input_ids=(
                torch.argmax(output.logits[:, -1, :], dim=-1, keepdim=True)
                if gen_new
                else as_gen[i].view(1, 1)
            ),
            past_key_values=output.past_key_values,
            use_cache=True,
        )

    torch.cuda.synchronize()
    decode_time = time() - start_time
    return prefill_time, decode_time


def main():
    enable_full_determinism(19260817)
    model_key = get_model_key()
    cache_ratio = int(os.environ.get("MOE_CACHE_RATIO", 0))
    print_log("use cache ratio", cache_ratio)

    model_dir = Path("../model" if cache_ratio == 0 else "../offload_model")
    in_dir = Path("../data/merged")
    out_dir = Path("../output/tp")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir / model_key,
        trust_remote_code=True,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation=model_config.loc[model_key, "attn"],
    ).eval()

    if cache_ratio != 0:
        for m in model.modules():
            if hasattr(m, "init_cache"):
                m.init_cache(cache_ratio)

    records = {}

    with torch.no_grad():
        for data_key, input_ids in load_file(in_dir / f"{model_key}.safetensors").items():
            torch.cuda.empty_cache()
            print_log("working on", data_key, "...")

            for i in range(5):
                _ = model(input_ids[-i].chunk(2)[1].unsqueeze(0).cuda(0))

            for flag, mode in enumerate(("old_doc", "new_gen")):
                key = f"{data_key}_{mode}"

                records[key] = torch.tensor(
                    [
                        measure_time(model, cur_input_ids, bool(flag))
                        for cur_input_ids in tqdm(input_ids[::16])
                    ]
                )

                prefill_time, decode_time = records[key].mean(dim=0).tolist()
                print_log(f"{data_key} {mode} prefill", prefill_time, "decode", decode_time)

    save_file(records, out_dir / f"{model_key}_{cache_ratio}.safetensors")


if __name__ == "__main__":
    main()
