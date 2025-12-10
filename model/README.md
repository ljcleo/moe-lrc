# REAL model preparation

We conduct experiments on the following MoE LLMs (sorted by model size):

- `powermoe`: PowerMoe-3B ([🤗 Repo](https://huggingface.co/ibm-research/PowerMoE-3b))
- `llamamoe`: LLaMA-MoE-v1-3.5B ([🤗 Repo](https://huggingface.co/llama-moe/LLaMA-MoE-v1-3_5B-4_16))
  - `llamamoesft`: LLaMA-MoE-v1-3.5B-SFT ([🤗 Repo](https://huggingface.co/llama-moe/LLaMA-MoE-v1-3_5B-4_16-sft))
- `olmoe`: OLMoE-1B-7B-0125 ([🤗 Repo](https://huggingface.co/allenai/OLMoE-1B-7B-0125))
  - `olmoesft`: OLMoE-1B-7B-0125-SFT ([🤗 Repo](https://huggingface.co/allenai/OLMoE-1B-7B-0125-SFT))
  - `olmoedpo`: OLMoE-1B-7B-0125-DPO ([🤗 Repo](https://huggingface.co/allenai/OLMoE-1B-7B-0125-DPO))
  - `olmoeins`: OLMoE-1B-7B-0125-Instruct ([🤗 Repo](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct))
- `switch`: SwitchTransformers-Base-128 ([🤗 Repo](https://huggingface.co/google/switch-base-128))
- `llamamoe2`: LLaMA-MoE-v2-3.8B ([🤗 Repo](https://huggingface.co/llama-moe/LLaMA-MoE-v2-3_8B-2_8-sft))
- `jetmoe`: JetMoE-8B ([🤗 Repo](https://huggingface.co/jetmoe/jetmoe-8b))
  - `jetmoesft`: JetMoE-8B-SFT ([🤗 Repo](https://huggingface.co/jetmoe/jetmoe-8b-sft))
  - `jetmoechat`: JetMoE-8B-Chat ([🤗 Repo](https://huggingface.co/jetmoe/jetmoe-8b-chat))
- `openmoe`: OpenMoE-8B ([🤗 Repo](https://huggingface.co/pingzhili/openmoe-8b-native-pt))
- `minicpm`: MiniCPM-MoE-8x2B ([🤗 Repo](https://huggingface.co/openbmb/MiniCPM-MoE-8x2B))
- `qwen`: Qwen1.5-MoE-A2.7B ([🤗 Repo](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B))
- `deepseek2`: DeepSeek-V2-Lite ([🤗 Repo](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite))
- `deepseek`: DeepSeekMoE ([🤗 Repo](https://huggingface.co/deepseek-ai/deepseek-moe-16b-base))
- `xverse`: XVERSE-MoE-A4.2B ([🤗 Repo](https://huggingface.co/xverse/XVERSE-MoE-A4.2B))
- `qwen3`: Qwen3-30B-A3B ([🤗 Repo](https://huggingface.co/Qwen/Qwen3-30B-A3B))
- `yuan`: Yuan2.0-M32 ([🤗 Repo](https://huggingface.co/IEITYuan/Yuan2-M32-hf))
- `phi`: Phi-3.5-MoE ([🤗 Repo](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct))
- `grin`: GRIN-MoE ([🤗 Repo](https://huggingface.co/microsoft/GRIN-MoE))
- `mixtral`: Mixtral-8x7B-v0.1 ([🤗 Repo](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1))
- `jamba`: Jamba-Mini-1.6 ([🤗 Repo](https://huggingface.co/ai21labs/AI21-Jamba-Mini-1.6))
- `nllb`: NLLB-MoE-54B ([🤗 Repo](https://huggingface.co/facebook/nllb-moe-54b))
- `qwen2`: Qwen2-57B-A14B ([🤗 Repo](https://huggingface.co/Qwen/Qwen2-57B-A14B))

We modify the original models' config files and modeling source codes to inject router logits logging code during inference; we only include these modified files in this repository to reduce repository size. Before running the Python scripts, please download the files not included here (e.g., tokenizers and model weights) from the original models' 🤗 repositories.

Optionally, you can create a [device map](https://huggingface.co/docs/accelerate/concept_guides/big_model_inference#designing-a-device-map) file named `device_map.json` for each model, and put it under the model's sub-folder. Once we detect it, we load model weights accordingly, otherwise we let `accelerate` to decide how model weights are loaded. We include an example for GRIN-MoE in this repository (`grin/device_map.json`; map keys depend on each model's modeling source code):

```json
{
    "model.embed_tokens": 0,
    "model.layers.0": 0,
    "model.layers.1": 0,
    "model.layers.2": 0,
    "model.layers.3": 0,
    "model.layers.4": 0,
    "model.layers.5": 0,
    "model.layers.6": 0,
    "model.layers.7": 0,
    "model.layers.8": 1,
    "model.layers.9": 1,
    "model.layers.10": 1,
    "model.layers.11": 1,
    "model.layers.12": 1,
    "model.layers.13": 1,
    "model.layers.14": 1,
    "model.layers.15": 1,
    "model.layers.16": 2,
    "model.layers.17": 2,
    "model.layers.18": 2,
    "model.layers.19": 2,
    "model.layers.20": 2,
    "model.layers.21": 2,
    "model.layers.22": 2,
    "model.layers.23": 2,
    "model.layers.24": 3,
    "model.layers.25": 3,
    "model.layers.26": 3,
    "model.layers.27": 3,
    "model.layers.28": 3,
    "model.layers.29": 3,
    "model.layers.30": 3,
    "model.layers.31": 3,
    "model.norm": 3,
    "lm_head": 3
}
```

**Next: [run scripts and notebooks](../src/README.md)**
