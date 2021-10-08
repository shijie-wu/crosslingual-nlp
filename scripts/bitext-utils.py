import os

import fire
from sacremoses import MosesTokenizer


def prepare():
    files = {}
    files["GALE"] = []
    dir = "intermediary/GALE_Arabic_flat"
    for mode in "train dev test".split():
        files["GALE"].append((f"{dir}/{mode}.en", f"{dir}/{mode}.ar", f"{dir}/{mode}"))
    files["MAIN"] = []
    dir = "intermediary"
    files["MAIN"].append(
        (
            f"{dir}/mt/train.src-tgt",
            f"{dir}/mt/train.src-tgt",
            f"{dir}/mt/train.src-tgt",
        )
    )
    files["MAIN"].append(
        (
            f"{dir}/ACE/english.src-tgt",
            f"{dir}/ACE/english.src-tgt",
            f"{dir}/ACE/english.src-tgt",
        )
    )
    for task in "ner-wiki ud2.7".split():
        for mode in "train dev test".split():
            input = f"{dir}/{task}/en.and_ar.helsinki_tok_nrns3.{mode}.text"
            output = f"{dir}/{task}/en.and_ar.helsinki_tok_nrns3.fast_align.{mode}"
            files["MAIN"].append((input, input, output))
    files["BETTER"] = []
    dir = "intermediary"
    files["BETTER"].append(
        (
            f"{dir}/mt/train.src-tgt",
            f"{dir}/mt/train.src-tgt",
            f"{dir}/mt/train.src-tgt",
        )
    )
    for mode in "analysis devtest train-0d train".split():
        files["BETTER"].append(
            (
                f"{dir}/better-abstract/{mode}.src",
                f"{dir}/better-abstract/helsinki/{mode}.src.trans",
                f"{dir}/fast_align-better-helsinki/{mode}",
            )
        )
    return files


def tokenize(input, output, lng):
    mt = MosesTokenizer(lang=lng)
    with open(input) as fp, open(output, "w") as out:
        for line in fp.readlines():
            text = line.strip()
            toks = mt.tokenize(text, escape=False)
            text = " ".join(toks)
            print(text, file=out)


def tokenize_all():
    for task in "ner-wiki ud2.7".split():
        for lng in "ar de es fr hi ru vi".split():
            for mode in "train dev test".split():
                input = f"intermediary/{task}/en.to_{lng}.helsinki_nrns3.{mode}.text"
                output = (
                    f"intermediary/{task}/en.to_{lng}.helsinki_tok_nrns3.{mode}.text"
                )
                assert os.path.isfile(input)
                print(input, output, lng)
                tokenize(input, output, lng)


def concat(dataset):
    files = prepare()
    assert dataset in files
    for src_fp, tgt_fp, out_fp in files[dataset]:
        print(f"SEP {src_fp} {tgt_fp} ||| {out_fp} SEP")
        with open(src_fp, "r") as sfp, open(tgt_fp, "r") as tfp:
            for i, (s, t) in enumerate(zip(sfp.readlines(), tfp.readlines())):
                s, t = s.strip(), t.strip()
                assert s and t
                if src_fp == tgt_fp:
                    assert s == t
                    assert " ||| " in s
                    print(s)
                else:
                    print(f"{s} ||| {t}")


def separate(file):
    assert os.path.isfile(f"{file}.text")
    assert os.path.isfile(f"{file}.align")
    print_text = None
    with open(f"{file}.text") as tfp, open(f"{file}.align") as afp:
        for i, (text, align) in enumerate(zip(tfp.readlines(), afp.readlines())):
            text = text.strip()
            align = align.strip()
            toks = text.split()
            if i == 0:
                assert toks[0] == "SEP"
                assert toks[-1] == "SEP"
            if toks[0] == "SEP" and toks[-1] == "SEP" and len(toks) == 6:
                output = toks[-2]
                if toks[1] != toks[2]:
                    otfp = open(f"{output}.text", "w")
                    oafp = open(f"{output}.align", "w")
                    print_text = True
                else:
                    oafp = open(f"{output}.align", "w")
                    print_text = False
                continue
            if print_text:
                print(text, file=otfp)
            print(align, file=oafp)


if __name__ == "__main__":
    fire.Fire()
