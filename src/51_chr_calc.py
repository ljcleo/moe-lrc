import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from sortedcontainers import SortedSet

from misc import get_global_config, get_model_key, model_config, print_log


class DS:
    def __init__(self, k: int):
        self.s = [(1, k + 1)]

    def query(self, p: int) -> int:
        k = self._find(p)
        if k == len(self.s):
            return p

        return self.s[k][0] if p == self.s[k][1] else p + (p >= self.s[k][0])

    def update(self, p: int, q: int):
        buf = []
        if q > 1:
            buf.append((1, q))

        k = self._find(p)

        if k < len(self.s) and self.s[k][0] <= p:
            if self.s[k][1] > p + 1:
                buf.append((p + 1, self.s[k][1]))
            k += 1

        buf.extend(self.s[k:])
        self.s = buf

    def _find(self, p: int) -> int:
        lp = 0
        rp = len(self.s) - 1

        while lp <= rp:
            m = (lp + rp) >> 1

            if self.s[m][1] < p:
                lp = m + 1
            else:
                rp = m - 1

        return lp


def calc_lru(seq: torch.Tensor, num_experts: int) -> torch.Tensor:
    prop = [0 for _ in range(num_experts)]
    queue = SortedSet()
    stat = [0 for _ in range(num_experts)]

    for i, x in enumerate(seq):
        x = x.item()
        t = prop[x]

        if t != 0:
            stat[queue.index((t, x))] += 1
            queue.remove((t, x))

        t = ~i
        prop[x] = t
        queue.add((t, x))

    return torch.tensor(stat).cumsum(0) / seq.shape[0]


def calc_lfu(seq: torch.Tensor, num_experts: int) -> torch.Tensor:
    prop = [(0, 0) for _ in range(num_experts)]
    queue = SortedSet()
    stat = [0 for _ in range(num_experts + 1)]
    gate = DS(num_experts)

    for i, x in enumerate(seq):
        x = x.item()
        f, t = prop[x]

        if f != 0:
            p = queue.index((f, t, x)) + 1
            stat[gate.query(p)] += 1
            queue.remove((f, t, x))
        else:
            p = num_experts + 1

        f -= 1
        t = ~i
        prop[x] = f, t
        queue.add((f, t, x))
        gate.update(p, queue.index((f, t, x)) + 1)

    return torch.tensor(stat).cumsum(0)[1:] / seq.shape[0]


def calc(index: int, topk: torch.Tensor, num_experts: int, out: torch.Tensor):
    out[index, 0] = calc_lru(topk[index], num_experts)
    out[index, 1] = calc_lfu(topk[index], num_experts)


def stat(
    topk: torch.Tensor, num_experts: int, bs_size: int, bs_up: int, bs_lp: int, rng: torch.Generator
) -> torch.Tensor:
    topk = topk.flatten(-2).share_memory_()
    pop = torch.zeros(topk.shape[0], 3, num_experts).share_memory_()

    with ProcessPoolExecutor(max_workers=round(os.cpu_count() * 0.9)) as pool:
        futures = [pool.submit(calc, i, topk, num_experts, pop) for i in range(topk.shape[0])]
    for future in futures:
        future.result()

    pop[:, 2] = (
        torch.stack([row.bincount(minlength=num_experts) for row in topk])[
            :, topk.ravel().bincount(minlength=num_experts).argsort(descending=True)
        ].cumsum(1)
        / topk.shape[1]
    )

    bs_len = pop.shape[0]

    bs_res = (
        pop[torch.randint(bs_len, (bs_size * bs_len,), generator=rng)]
        .unflatten(0, (bs_size, bs_len))
        .mean(dim=1)
        .sort(dim=0)
        .values
    )

    res = pop.mean(dim=0)
    res_l, res_u = res * 2 - bs_res[[bs_up, bs_lp]]
    return torch.stack([res_l, res, res_u], dim=1).contiguous()


def main():
    model_key = get_model_key()
    bs_size: int = get_global_config("num_bootstrap")
    ci_confidence: float = get_global_config("ci_confidence")
    bs_lp = round(bs_size * ((1 - ci_confidence) / 2))
    bs_up = bs_size - bs_lp

    root_dir = Path("../output")
    in_dir = root_dir / "topk"
    out_dir = root_dir / "chr"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = torch.manual_seed(hash(model_key))
    is_seq2seq = model_config.loc[model_key, "type"] == "seq2seq"
    num_layers = model_config.loc[model_key, "num_layers"]
    num_experts = model_config.loc[model_key, "num_experts"]
    results: dict[str, torch.Tensor] = {}
    buf: dict[str, list[torch.Tensor]] = {}
    tot_experts = {k: 0 for k in "de"}

    for layer_idx in range(num_layers):
        stem = f"{model_key}_{layer_idx:02d}"
        file = in_dir / f"{stem}.safetensors"

        if file.exists():
            print_log("working on", stem, "...")
            stage = "e" if is_seq2seq and layer_idx < num_layers // 2 else "d"

            for data_key, topk in load_file(file).items():
                results[f"{layer_idx:02d}_{data_key}"] = stat(
                    topk, num_experts, bs_size, bs_up, bs_lp, rng
                )

                buf_key = f"{stage}_{data_key}"
                topk = topk.to(torch.int16) + tot_experts[stage]

                if buf_key not in buf:
                    buf[buf_key] = [topk]
                else:
                    buf[buf_key].append(topk)

            tot_experts[stage] += num_experts
            print_log(stem, "finished.")

    if len(buf) > 0:
        print_log("working on", model_key, "...")

        for key, values in buf.items():
            stage = key.partition("_")[0]

            results[key] = stat(
                torch.cat(values, dim=-1), tot_experts[stage], bs_size, bs_up, bs_lp, rng
            )

        print_log(model_key, "finished.")

    if len(results) > 0:
        print_log("saving", model_key, "...")
        save_file(results, out_dir / f"{model_key}.safetensors")
        print_log(model_key, "saved.")


if __name__ == "__main__":
    main()
