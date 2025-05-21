from pathlib import Path

import orjson
import pyarrow as pa

from misc import get_data_keys


def main():
    data_keys = get_data_keys(from_arg=True)

    data_dir = Path("../data")
    in_dir = data_dir / "json"
    out_dir = data_dir / "arrow"
    out_dir.mkdir(parents=True, exist_ok=True)

    schema = pa.schema([pa.field(name="text", type=pa.string())])

    for data_key in data_keys:
        print("working on", data_key, "...")

        with pa.OSFile((out_dir / f"{data_key}.arrow").absolute().as_posix(), "wb") as sink:
            with pa.ipc.new_stream(sink=sink, schema=schema) as writer:
                buffer = []
                total = [0]

                def write_buffer():
                    writer.write(pa.record_batch([pa.array(buffer)], names=["text"]))
                    total[0] += len(buffer)
                    buffer.clear()
                    print(total[0])

                with (in_dir / f"{data_key}.jsonl").open(encoding="utf8") as f:
                    for line in f:
                        line = line.strip()

                        if len(line) > 0:
                            buffer.append(orjson.loads(line)["text"])
                            if len(buffer) >= 5000:
                                write_buffer()

                if len(buffer) > 0:
                    write_buffer()


if __name__ == "__main__":
    main()
