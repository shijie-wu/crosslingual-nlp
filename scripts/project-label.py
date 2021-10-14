import os
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Type

import ace_align_util
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
    final_path: Optional[str]
    if task == "better-abstract":
        final_path = f"{output_path}/silver-temp-{split}.json"
        better_temp: Dict[str, List] = {
            "spans_tgt": [],
            "offsets_tgt": [],
            "hspans_tgt": [],
            "hoffsets_tgt": [],
            "translation": [],
        }
    elif task == "ace":
        final_path = f"{output_path}/out.json"
    elif task == "muc":
        final_path = f"{output_path}/silver-{split}.json"
        o_data = OrderedDict()
    else:
        final_path = CLASS.get_file(output_path, name, split)
    if final_path is None:
        assert task == "ud"
        final_path = f"{output_path}/UD_{name}/{name}-ud-{split}.conllu"

    os.makedirs(os.path.dirname(final_path), exist_ok=True)

    if task == "ace":
        ace_alignments, ace_translations = [], []
        aligns = ace_align_util.load_aligns(bitext, alignment)
        for al in aligns:
            ace_alignments.append(al.align)
            ace_translations.append(al.tgt_tok)

        start_index = 0
        with open(final_path, "w") as fp:
            for example in CLASS.read_file(file_path, lang, split):
                silver = CLASS.project_label(
                    example,
                    ace_translations[
                        start_index : start_index + len(example["sentences"])
                    ],
                    ace_alignments[
                        start_index : start_index + len(example["sentences"])
                    ],
                )
                start_index += len(example["sentences"])
                CLASS.write_example(silver, fp)

    else:
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
                elif task == "muc":
                    o_data.update(silver)
                else:
                    better_temp["spans_tgt"].append(silver["spans_tgt"])
                    better_temp["offsets_tgt"].append(silver["offsets_tgt"])
                    better_temp["hspans_tgt"].append(silver["hspans_tgt"])
                    better_temp["hoffsets_tgt"].append(silver["hoffsets_tgt"])
                    better_temp["translation"].append(silver["translation"])

            if "better" in task:
                better_temp["file_path"] = [file_path]
                CLASS.write_example(better_temp, fp)
            if task == "muc":
                CLASS.write_example(o_data, fp)


if __name__ == "__main__":
    fire.Fire(main)
