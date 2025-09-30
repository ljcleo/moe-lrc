# Data preparation

We generate our sample data from these datasets:

- Generic corpora:
  - `RedPajama-Data-1T-Sample` ([🤗 Repo](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample))
- Downstream application:
  - `arena-human-preference-140k` ([🤗 Repo](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k))
  - `OpenMathInstruct-2` ([🤗 Repo](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2))
  - `OpenCodeInstruct` ([🤗 Repo](https://huggingface.co/datasets/nvidia/OpenCodeInstruct))
  - `OpenScienceReasoning-2` ([🤗 Repo](https://huggingface.co/datasets/nvidia/OpenScienceReasoning-2))

We convert all plain text data from the original datasets (with necessary preprocessing) into the arrow format; due to large file sizes, we release them in the [Releases](https://github.com/ljcleo/moe-lrc/releases) tab. Before running the data preparation scripts (`0?-*.py`), please create a sub-folder named `arrow` in this folder, download the released archive, and extract the arrow files into the sub-folder.

**Next: [prepare models](../model/README.md)**
