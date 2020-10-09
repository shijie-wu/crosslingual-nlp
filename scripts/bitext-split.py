import fire


def write(text, align, prefix):
    with open(f"{prefix}.text", "w") as tfp, open(f"{prefix}.align", "w") as afp:
        for t, a in zip(text, align):
            if t == "1 ||| 1":
                continue
            print(t, file=tfp)
            print(a, file=afp)


def main(text_fp, align_fp, nb_val, prefix):
    with open(text_fp, "r") as tfp:
        text = [line.strip() for line in tfp.readlines()]
    with open(align_fp, "r") as afp:
        align = [line.strip() for line in afp.readlines()]

    assert len(text) == len(align)
    text_val, text_trn = text[:nb_val], text[nb_val:]
    align_val, align_trn = align[:nb_val], align[nb_val:]
    write(text_trn, align_trn, f"{prefix}.trn")
    write(text_val, align_val, f"{prefix}.val")


if __name__ == "__main__":
    fire.Fire(main)
