import json
from pathlib import Path

import pandas as pd
from fire import Fire


def pprint(df):
    lines = df.to_csv(float_format="%.4f")
    if "_las," and "_uas," in lines:
        print(
            "\n".join(
                [
                    line
                    for line in lines.split("\n")
                    if "mean" in line or "_las," in line
                ]
            )
        )
        print()
        print(
            "\n".join(
                [
                    line
                    for line in lines.split("\n")
                    if "mean" in line or "_uas," in line
                ]
            )
        )
    else:
        print(lines)


class Main:
    def read_jsonl(self, file):
        data = []
        with open(file) as fp:
            for line in fp.readlines():
                line = line.strip()
                data.append(json.loads(line))

        val, tst = data[:-1], data[-1]
        i, k = {v: (i, k) for i, d in enumerate(val) for k, v in d.items()}[
            tst["select"]
        ]
        tst.update(val[i])
        tst["select_criterion"] = k
        tst["model"] = str(Path(file).parent)
        tst = {k: v for k, v in tst.items() if "loss" not in k}
        tst = {k: v for k, v in tst.items() if "lem" not in k}
        tst = {k: v for k, v in tst.items() if "uem" not in k}
        tst = {k: v for k, v in tst.items() if "val" not in k}
        return tst

    def best(self, directory, ret=False):
        directory = Path(directory)
        assert directory.is_dir()

        results = [self.read_jsonl(fp) for fp in directory.glob("**/*.jsonl")]
        df = pd.DataFrame(results)
        del df["model"]
        best_model = df.loc[df["select"].argmax()]
        if ret:
            return best_model
        else:
            pprint(best_model)

    def mean_std(self, directory, ret=False):
        directory = Path(directory)
        assert directory.is_dir()

        results = [self.read_jsonl(fp) for fp in directory.glob("**/*.jsonl")]
        df = pd.DataFrame(results)
        del df["model"]
        final = pd.DataFrame(data={"mean": df.mean(), "std": df.std()})
        if ret:
            return final
        else:
            pprint(final)


if __name__ == "__main__":
    Fire(Main)
