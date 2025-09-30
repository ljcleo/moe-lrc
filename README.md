# Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models

This is the codebase for:
<p align=center><i>Not All Models Suit Expert Offloading:<br>On <b>Local Routing Consistency</b> of Mixture-of-Expert Models</i></p>
<p align=center>[<a href="https://arxiv.org/abs/2505.16056">📄Paper</a>] • [<a href="https://github.com/ljcleo/moe-lrc">💻Code</a>]</p>

![Sample routing score of GRIN-MoE and Jamba-Mini-1.6](plot/sample.png)

## Requirements

We recommend using [uv](https://docs.astral.sh/uv/) to setup a Python environment. Due to an issue of `causal-conv1d`, you might need to run `uv venv` and `uv pip install setuptools torch` before running `uv sync`.

## Usage

1. **[Download raw data files](data/README.md)**
2. **[Download model files](model/README.md)**
3. **[Run scripts and notebooks!](src/README.md)**

## Cite

```bibtex
@misc{liang2025modelssuitexpertoffloading,
      title={Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models}, 
      author={Jingcong Liang and Siyuan Wang and Miren Tian and Yitong Li and Duyu Tang and Zhongyu Wei},
      year={2025},
      eprint={2505.16056},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.16056}, 
}
```
