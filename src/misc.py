from argparse import ArgumentParser
from datetime import datetime
from typing import Any

import orjson
import pandas as pd

with open("../config.json", "rb") as f:
    global_config: dict[str, Any] = orjson.loads(f.read())


def get_global_config(key: str) -> Any | None:
    return global_config.get(key, None)


def read_config(sub):
    df = pd.read_json(f"../{sub}/config.jsonl", lines=True)
    df["key"] = df["key"].astype(pd.CategoricalDtype(df["key"].to_list(), ordered=True))
    return df.set_index("key")


data_config = read_config("data")
model_config = read_config("model")


def get_data_keys(from_arg: bool = False) -> list[str]:
    data_keys: list[str] = data_config.index.to_list()

    if from_arg:
        parser = ArgumentParser()
        parser.add_argument("datasets", nargs="*", choices=data_keys, metavar="dataset")
        arg_data_keys: list[str] | None = parser.parse_args().datasets

        if arg_data_keys is not None and len(arg_data_keys) > 0:
            data_keys = arg_data_keys

    return data_keys


def get_model_keys(from_arg: bool = False) -> list[str]:
    model_keys: list[str] = model_config.index.to_list()

    if from_arg:
        parser = ArgumentParser()
        parser.add_argument("models", nargs="*", choices=model_keys, metavar="model")
        arg_model_keys: list[str] | None = parser.parse_args().models

        if arg_model_keys is not None and len(arg_model_keys) > 0:
            model_keys = arg_model_keys

    return model_keys


def get_model_key() -> str:
    parser = ArgumentParser()
    parser.add_argument("model", choices=get_model_keys(), metavar="model")
    return parser.parse_args().model


def get_data_and_model_keys(from_arg: bool = False) -> tuple[list[str], list[str]]:
    data_keys: list[str] = get_data_keys()
    model_keys: list[str] = get_model_keys()

    if from_arg:
        parser = ArgumentParser()
        parser.add_argument("-d", "--datasets", nargs="*", choices=data_keys, metavar="dataset")
        parser.add_argument("-m", "--models", nargs="*", choices=model_keys, metavar="model")
        args = parser.parse_args()
        arg_data_keys: list[str] | None = args.datasets
        arg_model_keys: list[str] | None = args.models

        if arg_data_keys is not None and len(arg_data_keys) > 0:
            data_keys = arg_data_keys
        if arg_model_keys is not None and len(arg_model_keys) > 0:
            model_keys = arg_model_keys

    return data_keys, model_keys


def print_log(*msg: Any) -> None:
    print(datetime.now().strftime("[%Y/%m/%d %H:%M:%S]"), *msg)
