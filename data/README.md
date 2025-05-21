# Data Preparation

We generate our sample data from `RedPajama-Data-1T-Sample` ([🤗 Repo](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample)).

Before running the data preparation scripts (`0?-*.py`), please create a sub-folder named `json` in this folder, and download raw JSON-line files from the 🤗 Repository:

- `arxiv.jsonl`: <https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample/resolve/main/arxiv_sample.jsonl>
- `book.jsonl`: <https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample/resolve/main/book_sample.jsonl>
- `c4.jsonl`: <https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample/resolve/main/c4_sample.jsonl>
- `commoncrawl.jsonl`: <https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample/resolve/main/cc_2023-06_sample.jsonl>
- `github.jsonl`: <https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample/resolve/main/github_sample.jsonl>
- `stackexchange.jsonl`: <https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample/resolve/main/stackexchange_sample.jsonl>
- `wikipedia.jsonl`: <https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample/resolve/main/wikipedia_sample.jsonl>

**Next: [prepare models](../model/README.md)**
