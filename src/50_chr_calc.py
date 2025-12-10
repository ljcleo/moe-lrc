import os
from bisect import bisect_left
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from sortedcontainers import SortedSet
from tqdm import tqdm, trange

from misc import get_global_config, get_model_key, model_config, print_log, stable_hash


def calc_lru(seq: torch.Tensor, num_experts: int) -> torch.Tensor:
    prop = [0 for _ in range(num_experts)]
    queue = SortedSet()
    stat = [0 for _ in range(num_experts)]
    uni: int = 0
    ts: int = 0

    for batch in seq:
        x: int

        for expert in batch:
            x = expert.item()

            if prop[x] != 0:
                stat[queue.index(prop[x])] += 1
            elif uni < num_experts:
                stat[uni] += 1
                uni += 1

        for expert in batch:
            x = expert.item()
            if prop[x] != 0:
                queue.remove(prop[x])

            ts -= 1
            prop[x] = ts
            queue.add(prop[x])

    return torch.tensor(stat).cumsum(0) / seq.numel()


def calc_lfu(seq: torch.Tensor, num_experts: int) -> torch.Tensor:
    prop: list[tuple[int, int]] = [(0, 0) for _ in range(num_experts)]
    stat: list[int] = [0 for _ in range(num_experts + 1)]
    gate: list[tuple[int, int]] = [(1, num_experts + 1)]
    queue: SortedSet = SortedSet()
    uni: int = 0
    ts: int = 0

    for batch in seq:
        x: int

        for expert in batch:
            x = expert.item()

            if prop[x][0] != 0:
                p: int = queue.index(prop[x]) + 1
                k: int = bisect_left(gate, p, key=lambda x: x[1])

                if k == len(gate):
                    stat[p] += 1
                elif p == gate[k][1]:
                    stat[gate[k][0]] += 1
                else:
                    stat[p + (p >= gate[k][0])] += 1

            elif uni < num_experts:
                stat[uni + 1] += 1
                uni += 1

        for expert in batch:
            x = expert.item()

            if prop[x][0] != 0:
                p = queue.index(prop[x]) + 1
                queue.remove(prop[x])
            else:
                p = num_experts + 1

            ts -= 1
            prop[x] = prop[x][0] - 1, ts
            queue.add(prop[x])
            buf: list[tuple[int, int]] = []

            q: int = queue.index(prop[x]) + 1
            if q > 1:
                buf.append((1, q))

            k = bisect_left(gate, p, key=lambda x: x[1])

            if k < len(gate) and gate[k][0] <= p:
                if gate[k][1] > p + 1:
                    buf.append((p + 1, gate[k][1]))

                k += 1

            buf.extend(gate[k:])
            gate = buf

    return torch.tensor(stat).cumsum(0)[1:] / seq.numel()


def calc_beladi(seq: torch.Tensor, num_experts: int) -> torch.Tensor:
    ts: int = 0
    vis_t: list[deque[int]] = [deque() for _ in range(num_experts)]
    x: int

    for batch in seq:
        for expert in batch:
            x = expert.item()
            ts += 1
            vis_t[x].append(ts)

    for x in range(num_experts):
        ts += 1
        vis_t[x].append(ts)

    prop: list[int] = [0 for _ in range(num_experts)]
    stat: list[int] = [0 for _ in range(num_experts)]
    rank: list[int] = [num_experts for _ in range(num_experts)]
    prv: list[int] = list(range(num_experts + 1))
    nxt: list[int] = list(range(num_experts + 1))
    hdp: list[int] = list(range(num_experts + 1))
    hdn: list[int] = list(range(num_experts + 1))
    uni: int = 0

    for batch in seq:
        p: int = nxt[num_experts]
        c: int = 0

        while p != num_experts:
            rank[p] = c
            p = nxt[p]
            c += 1

        for expert in batch:
            x = expert.item()
            pos: int = rank[x]

            if pos == num_experts:
                stat[uni] += 1
                uni += 1
            else:
                stat[pos] += 1

        for expert in batch:
            x = expert.item()
            q: int

            if nxt[x] == x:
                q = num_experts
            else:
                if hdn[x] != -1:
                    q = x
                else:
                    p1: int = x
                    p2: int = nxt[x]

                    while hdn[p1] == -1 and hdn[p2] == -1:
                        p1 = prv[p1]
                        p2 = nxt[p2]

                    p = hdp[p2] if hdn[p1] == -1 else p1
                    q = hdn[p]
                    hdn[p] = x
                    hdp[x] = p
                    hdn[x] = q
                    hdp[q] = x
                    q = x

            p = hdn[num_experts]

            while p != num_experts and p != q and hdn[p] != num_experts and hdn[p] != q:
                c = prv[hdn[p]]
                r: int = hdn[p]

                while hdn[r] != num_experts and hdn[r] != q and prop[prv[hdn[r]]] > prop[c]:
                    r = hdn[r]

                d1: int = r
                d2: int = hdn[r]

                while (
                    hdn[nxt[d1]] == -1
                    and prop[nxt[d1]] > prop[c]
                    and hdn[prv[d2]] == -1
                    and prop[prv[d2]] < prop[c]
                ):
                    d1 = nxt[d1]
                    d2 = prv[d2]

                d: int = prv[d2] if hdn[nxt[d1]] == -1 and prop[nxt[d1]] > prop[c] else d1
                a: int = prv[c]
                b: int = nxt[c]
                nxt[a] = b
                prv[b] = a

                if hdn[c] != -1:
                    s: int = hdp[c]
                    t: int = hdn[c]
                    hdn[s] = t
                    hdp[t] = s
                    hdp[c] = hdn[c] = -1

                if a != num_experts and b != num_experts and prop[a] > prop[b]:
                    s: int = hdp[b]
                    t: int = hdn[b]
                    hdn[s] = t
                    hdp[t] = s
                    hdp[b] = hdn[b] = -1

                    if r == b:
                        r = s

                a = nxt[d]
                nxt[d] = c
                prv[c] = d
                nxt[c] = a
                prv[a] = c
                p = r

            if nxt[x] != x:
                a: int = prv[x]
                b: int = nxt[x]
                nxt[a] = b
                prv[b] = a
                c = hdp[x]
                d: int = hdn[x]

                if d == b:
                    if a != num_experts and b != num_experts and prop[a] > prop[b]:
                        a = hdn[d]
                        hdn[c] = a
                        hdp[a] = c
                        hdp[d] = hdn[d] = -1
                    else:
                        hdn[c] = d
                        hdp[d] = c
                else:
                    if a != num_experts and b != num_experts and prop[a] > prop[b]:
                        hdn[c] = d
                        hdp[d] = c
                    else:
                        hdn[c] = b
                        hdp[b] = c
                        hdn[b] = d
                        hdp[d] = b

                hdp[x] = hdn[x] = -1

            q = nxt[num_experts]
            nxt[num_experts] = x
            prv[x] = num_experts
            nxt[x] = q
            prv[q] = x
            hdn[num_experts] = x
            hdp[x] = num_experts

            vis_t[x].popleft()
            prop[x] = -vis_t[x][0]

            if q != num_experts and prop[x] > prop[q]:
                c = hdn[q]
                hdn[x] = c
                hdp[c] = x
                hdp[q] = hdn[q] = -1
            else:
                hdn[x] = q
                hdp[q] = x

    return torch.tensor(stat).cumsum(0) / seq.numel()


def calc(index: int, topk: torch.Tensor, num_experts: int, out: torch.Tensor):
    out[index, 0] = calc_lru(topk[index], num_experts)
    out[index, 1] = calc_lfu(topk[index], num_experts)
    out[index, 2] = calc_beladi(topk[index], num_experts)


def stat(
    topk: torch.Tensor,
    num_experts: int,
    bs_size: int,
    bs_up: int,
    bs_lp: int,
    rng: torch.Generator,
    pbar: tqdm | None = None,
) -> torch.Tensor:
    topk = topk.share_memory_()
    pop = torch.zeros(topk.shape[0], 3, num_experts).share_memory_()
    max_workers = 32
    cpu_count = os.cpu_count()

    if cpu_count is not None:
        max_workers = round(cpu_count * 0.9)

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for future in [pool.submit(calc, i, topk, num_experts, pop) for i in range(topk.shape[0])]:
            future.result()
            if pbar is not None:
                pbar.update(1)

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

    rng = torch.manual_seed(stable_hash(model_key))
    is_seq2seq = model_config.loc[model_key, "type"] == "seq2seq"
    num_layers = model_config.loc[model_key, "num_layers"]
    num_experts = model_config.loc[model_key, "num_experts"]
    results: dict[str, torch.Tensor] = {}
    buf: dict[str, list[torch.Tensor]] = {}
    tot_experts = {k: 0 for k in "de"}
    print_log("running layers of", model_key, "...")

    for layer_idx in trange(num_layers):
        stem = f"{model_key}_{layer_idx:02d}"
        file = in_dir / f"{stem}.safetensors"

        if file.exists():
            # print_log("working on", stem, "...")
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
            # print_log(stem, "finished.")

    if len(buf) > 0:
        print_log("running model", model_key, "...")

        with tqdm(total=sum(values[0].shape[0] for values in buf.values())) as pbar:
            for key, values in buf.items():
                stage = key.partition("_")[0]

                results[key] = stat(
                    torch.cat(values, dim=-1),
                    tot_experts[stage],
                    bs_size,
                    bs_up,
                    bs_lp,
                    rng,
                    pbar=pbar,
                )

        # print_log(model_key, "finished.")

    if len(results) > 0:
        print_log("saving", model_key, "...")
        save_file(results, out_dir / f"{model_key}.safetensors")
        print_log(model_key, "saved.")


if __name__ == "__main__":
    main()
