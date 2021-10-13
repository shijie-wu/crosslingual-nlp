import os
import sys
from typing import Dict, Optional, Type

import fire

sys.path.append("src")

from dataset import Dataset, ParsingDataset, WikiAnnNER  # noqa: E402


def main(
    task: str,
    path: str,
    lang: str,
    split: str,
    bitext: str,
    alignment: str,
    output_path: str,
    name: str,
):
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
    final_path: Optional[str]
    if task == "ud" or task == "wikiann":
        final_path = CLASS.get_file(output_path, name, split)
    if final_path is None:
        assert task == "ud"
        final_path = f"{output_path}/UD_{name}/{name}-ud-{split}.conllu"

    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    if task == "ud" or task == "wikiann":
        with open(bitext) as tfp, open(alignment) as afp, open(final_path, "w") as fp:
            for example, bitext, alignment in zip(
                CLASS.read_file(file_path, lang, split),
                tfp.readlines(),
                afp.readlines(),
            ):
                src_text, tgt_text = bitext.strip().split(" ||| ")
                mapping = [
                    tuple([int(i) for i in x.split("-")])
                    for x in alignment.strip().split(" ")
                ]
                if task == "ud" or task == "wikiann":
                    # does not work for better because of &apos;s
                    assert (
                        " ".join(example["sent"]) == src_text
                        or bitext.strip() == "1 ||| 1"
                    )

                silver = CLASS.project_label(example, tgt_text.split(), mapping)
                if task == "ud" or task == "wikiann":
                    CLASS.write_example(silver, fp)


if __name__ == "__main__":
    fire.Fire(main)
