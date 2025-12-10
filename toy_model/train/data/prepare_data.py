import os
import sys

import numpy as np
from megatron.core.datasets import indexed_dataset


def main():
    data_dir = sys.argv[1]
    with open("all.txt", encoding="utf8") as f:
        input_files = [os.path.join(data_dir, r.strip()) for r in f]

    builder = indexed_dataset.IndexedDatasetBuilder("train_text_document.bin", dtype=np.uint16)
    total_docs = 0

    for file in input_files:
        print(f"Processing {file} ...")

        with open(file, "rb") as f:
            data = f.read(1 << 25)
        if len(data) == 0:
            continue

        arr = np.frombuffer(data, dtype=np.uint16)
        start = 0

        for pos in (arr == 50279).nonzero()[0]:
            builder.add_document(arr[start : pos + 1].tolist(), [pos + 1 - start])
            start = pos + 1
            total_docs += 1

            if total_docs % 100000 == 0:
                print(f"Added {total_docs} documents ...", file=sys.stderr)

    builder.finalize("train_text_document.idx")
    print(f"Done. Wrote {total_docs} documents.")


if __name__ == "__main__":
    main()
