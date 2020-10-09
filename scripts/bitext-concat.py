import fire


def main(src_fp, tgt_fp):
    with open(src_fp, "r") as sfp, open(tgt_fp, "r") as tfp:
        for i, (s, t) in enumerate(zip(sfp.readlines(), tfp.readlines())):
            s, t = s.strip(), t.strip()
            # assert s and t
            # print(f'{s} ||| {t}')
            if s and t:
                print(f"{s} ||| {t}")
            else:
                print("1 ||| 1")


if __name__ == "__main__":
    fire.Fire(main)
