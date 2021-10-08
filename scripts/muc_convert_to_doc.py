# type: ignore
import sys

import fire

sys.path.append("src")

from dataset import MUCDataset  # noqa: E402


def main(
    path: str,
    lang: str,
    split: str,
    bitext: str,
    alignment: str,
    output_bitext: str,
    output_alignment: str,
):
    o_a = open(output_alignment, "w")
    o_b = open(output_bitext, "w")
    b_list = []
    a_list = []
    src = []
    # mt = MosesTokenizer(lang="en")
    file_path = MUCDataset.get_file(path, lang, split)
    with open(bitext, "r") as tfp:
        for line in tfp:
            src_text, tgt_text = line.strip().split(" ||| ")
            b_list.append((src_text, tgt_text))
            src.append(src_text.split())

    with open(alignment, "r") as tfp:
        for line in tfp:
            mapping = [
                tuple([int(i) for i in x.split("-")]) for x in line.strip().split(" ")
            ]
            a_list.append(mapping)
    for example in MUCDataset.read_file(file_path, lang, split):
        trnslt = ""
        als = ""
        start_src = 0
        start_trg = 0
        for i in range(len(example["sent"])):
            trnslt += b_list[i][1] + " "
            for j in a_list[i]:
                als += str(start_src + j[0]) + "-" + str(start_trg + j[1]) + " "
            start_src += len(src[i])
            start_trg += len(b_list[i][1].split())
        o_a.write(als + "\n")
        o_b.write(" ".join(example["doc"]) + " ||| " + trnslt + "\n")
        src = src[len(example["sent"]) :]
        a_list = a_list[len(example["sent"]) :]
        b_list = b_list[len(example["sent"]) :]


if __name__ == "__main__":
    fire.Fire(main)
