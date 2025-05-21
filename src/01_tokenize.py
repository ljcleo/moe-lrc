from functools import partial
from pathlib import Path

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from misc import get_data_and_model_keys, get_global_config


def tokenize(tokenizer, examples):
    tokens = tokenizer(examples["text"], truncation=False, return_attention_mask=False)["input_ids"]

    for token in tokens:
        if token[-1] != tokenizer.eos_token_id:
            token.append(tokenizer.eos_token_id)

    return {"k": tokens}


def main():
    data_keys, model_keys = get_data_and_model_keys(from_arg=True)

    data_dir = Path("../data")
    model_dir = Path("../model")
    in_dir = data_dir / "arrow"
    out_dir = data_dir / "tokenized"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizers = {}
    num_samples = get_global_config("num_samples")
    sample_length = get_global_config("sample_length")

    for model_key in model_keys:
        tokenizer = AutoTokenizer.from_pretrained(model_dir / model_key, trust_remote_code=True)
        assert sample_length <= tokenizer.model_max_length
        tokenizers[model_key] = tokenizer

    for data_key in data_keys:
        print("current dataset", data_key)
        dataset = Dataset.from_file((in_dir / f"{data_key}.arrow").as_posix())
        max_take = len(dataset) // 32

        for model_key, tokenizer in tokenizers.items():
            print("working with", model_key, "...")
            is_nllb = model_key == "nllb"

            output = []
            buffer = []

            with tqdm(total=num_samples) as pbar:
                num_tokenized = 0

                while len(output) < num_samples:
                    num_take = min(max_take, len(dataset) - num_tokenized)
                    assert num_take > 0

                    tokens = (
                        dataset.skip(num_tokenized)
                        .take(num_take)
                        .map(partial(tokenize, tokenizer), batched=True, remove_columns=["text"])
                    )

                    num_tokenized += num_take

                    for row in tokens:
                        buffer.extend(row["k"][is_nllb and len(buffer) > 0 :])

                        while len(output) < num_samples and len(buffer) >= sample_length:
                            output.append(buffer[:sample_length])
                            buffer = (buffer[:1] if is_nllb else []) + buffer[sample_length:]
                            pbar.update(1)

                        if len(output) >= num_samples:
                            break

            assert len(output) == num_samples
            for sample in output:
                assert len(sample) == sample_length

            torch.save(torch.tensor(output), out_dir / f"{model_key}_{data_key}.pt")


if __name__ == "__main__":
    main()
