import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def split_qkv(qkv, head_size):
    hidden_size = qkv.shape[-1]
    head_num = hidden_size // head_size
    qkv_total_dim = qkv.shape[0] // head_size
    num_query_groups = (qkv_total_dim - head_num) // 2
    heads_per_group = head_num // num_query_groups
    total_heads_per_group = heads_per_group + 2

    qkv_reshaped = qkv.view(qkv_total_dim, head_size, hidden_size)

    q_slice = torch.cat(
        [
            torch.arange(total_heads_per_group * i, total_heads_per_group * i + heads_per_group)
            for i in range(num_query_groups)
        ]
    )

    k_slice = torch.arange(total_heads_per_group - 2, qkv_total_dim, total_heads_per_group)
    v_slice = torch.arange(total_heads_per_group - 1, qkv_total_dim, total_heads_per_group)

    q = qkv_reshaped[q_slice]
    k = qkv_reshaped[k_slice]
    v = qkv_reshaped[v_slice]

    assert q.numel() + k.numel() + v.numel() == qkv.numel(), (
        f"QKV weights are not correctly merged, {q.shape=}, {k.shape=}, {v.shape=}, {qkv.shape=}"
    )

    q = q.reshape(-1, hidden_size)
    k = k.reshape(-1, hidden_size)
    v = v.reshape(-1, hidden_size)
    return q, k, v


def work(in_dir, tmpl_dir, out_dir):
    old = torch.load(
        in_dir / "mp_rank_00" / "model_optim_rng.pt", weights_only=False, map_location="cpu"
    )["model"]
    new = {}

    new["model.embed_tokens.weight"] = old["embedding.word_embeddings.weight"]
    x = 0

    while f"decoder.layers.{x}.self_attention.linear_qkv.layer_norm_weight" in old:
        new[f"model.layers.{x}.input_layernorm.weight"] = old[
            f"decoder.layers.{x}.self_attention.linear_qkv.layer_norm_weight"
        ]

        (
            new[f"model.layers.{x}.self_attn.q_proj.weight"],
            new[f"model.layers.{x}.self_attn.k_proj.weight"],
            new[f"model.layers.{x}.self_attn.v_proj.weight"],
        ) = split_qkv(
            old[f"decoder.layers.{x}.self_attention.linear_qkv.weight"],
            old[f"decoder.layers.{x}.self_attention.q_layernorm.weight"].shape[0],
        )

        new[f"model.layers.{x}.self_attn.q_norm.weight"] = old[
            f"decoder.layers.{x}.self_attention.q_layernorm.weight"
        ]

        new[f"model.layers.{x}.self_attn.k_norm.weight"] = old[
            f"decoder.layers.{x}.self_attention.k_layernorm.weight"
        ]

        new[f"model.layers.{x}.self_attn.o_proj.weight"] = old[
            f"decoder.layers.{x}.self_attention.linear_proj.weight"
        ]

        if f"decoder.layers.{x}.mlp.linear_fc1.layer_norm_weight" in old:
            new[f"model.layers.{x}.post_attention_layernorm.weight"] = old[
                f"decoder.layers.{x}.mlp.linear_fc1.layer_norm_weight"
            ]

            (
                new[f"model.layers.{x}.mlp.gate_proj.weight"],
                new[f"model.layers.{x}.mlp.up_proj.weight"],
            ) = torch.chunk(old[f"decoder.layers.{x}.mlp.linear_fc1.weight"], 2, dim=0)

            new[f"model.layers.{x}.mlp.down_proj.weight"] = old[
                f"decoder.layers.{x}.mlp.linear_fc2.weight"
            ]
        else:
            new[f"model.layers.{x}.post_attention_layernorm.weight"] = old[
                f"decoder.layers.{x}.pre_mlp_layernorm.weight"
            ]

            new[f"model.layers.{x}.mlp.gate.weight"] = old[f"decoder.layers.{x}.mlp.router.weight"]
            y = 0

            while f"decoder.layers.{x}.mlp.experts.linear_fc1.weight{y}" in old:
                (
                    new[f"model.layers.{x}.mlp.experts.{y}.gate_proj.weight"],
                    new[f"model.layers.{x}.mlp.experts.{y}.up_proj.weight"],
                ) = torch.chunk(
                    old[f"decoder.layers.{x}.mlp.experts.linear_fc1.weight{y}"], 2, dim=0
                )

                new[f"model.layers.{x}.mlp.experts.{y}.down_proj.weight"] = old[
                    f"decoder.layers.{x}.mlp.experts.linear_fc2.weight{y}"
                ]

                y += 1

            if f"decoder.layers.{x}.mlp.shared_experts.linear_fc1.weight" in old:
                (
                    new[f"model.layers.{x}.mlp.shared_experts.gate_proj.weight"],
                    new[f"model.layers.{x}.mlp.shared_experts.up_proj.weight"],
                ) = torch.chunk(
                    old[f"decoder.layers.{x}.mlp.shared_experts.linear_fc1.weight"], 2, dim=0
                )

                new[f"model.layers.{x}.mlp.shared_experts.down_proj.weight"] = old[
                    f"decoder.layers.{x}.mlp.shared_experts.linear_fc2.weight"
                ]

        x += 1

    new["model.norm.weight"] = old["decoder.final_layernorm.weight"]
    new["lm_head.weight"] = old["output_layer.weight"]

    model = AutoModelForCausalLM.from_config(
        AutoConfig.from_pretrained(tmpl_dir, trust_remote_code=True), trust_remote_code=True
    )

    model.load_state_dict(new)
    model.save_pretrained(out_dir)


def main():
    exp = sys.argv[1]
    in_dir = Path(f"./experiments/{exp}")
    tmpl_dir = Path(f"./hf/models/{exp}")
    out_dir = Path("./hf/checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=10) as pool:
        for future in [
            pool.submit(
                work, in_dir / f"iter_00{i + 1:02d}000", tmpl_dir, out_dir / f"{exp}_{i + 1:02d}"
            )
            for i in range(10)
        ]:
            future.result()


if __name__ == "__main__":
    main()
