import sys
from typing import Dict, Type

import fire

sys.path.append("src")
from dataset import Dataset, ParsingDataset, WikiAnnNER  # noqa: E402


def main(task: str, path: str, lang: str, split: str):

    MAPPING: Dict[str, Type[Dataset]] = {
        "wikiann": WikiAnnNER,
        "ud": ParsingDataset,
    }
    assert task in MAPPING
    CLASS = MAPPING[task]

    file_path = CLASS.get_file(path, lang, split)
    if file_path is None:
        print("Empty file path")
        exit()
    for example in CLASS.read_file(file_path, lang, split):
        if task == "ud" or task == "wikiann":
            print(" ".join(example["sent"]))


if __name__ == "__main__":
    fire.Fire(main)
