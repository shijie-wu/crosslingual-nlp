import sys
from typing import Dict, Type

import fire

sys.path.append("src")
from dataset import (  # noqa: E402
    ACEDataset,
    BetterDataset,
    Dataset,
    MUCDataset,
    ParsingDataset,
    WikiAnnNER,
)


def main(task: str, path: str, lang: str, split: str):

    MAPPING: Dict[str, Type[Dataset]] = {
        "wikiann": WikiAnnNER,
        "ud": ParsingDataset,
        "better-abstract": BetterDataset,
        "ace": ACEDataset,
        "muc": MUCDataset,
    }
    assert task in MAPPING
    CLASS = MAPPING[task]

    file_path = CLASS.get_file(path, lang, split)
    if file_path is None:
        print("Empty file path")
        exit()
    for example in CLASS.read_file(file_path, lang, split):
        if task == "ace":
            for s in example["sentences"]:
                print(" ".join(s["tokens"]))
        elif task == "muc":
            for i in example["sent"]:
                print(" ".join(i))
        else:
            print(" ".join(example["sent"]))


if __name__ == "__main__":
    fire.Fire(main)
