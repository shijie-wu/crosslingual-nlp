import os
import sys
from glob import glob

import torch
import torch.nn as nn
from fire import Fire
from tqdm import tqdm


class Main:
    def load_checkpoint(self, filepath):
        if "src" not in sys.path:
            sys.path.append("src")
        if "aligner" not in sys.modules:
            import aligner
        else:
            aligner = sys.modules["aligner"]

        return aligner.Aligner.load_from_checkpoint(filepath)

    def single(self, ckpt, dumpdir):
        ckpt = glob(ckpt, recursive=True)
        assert len(ckpt) == 1, ckpt
        aligner = self.load_checkpoint(ckpt[0])
        os.makedirs(dumpdir, exist_ok=True)
        aligner.model.save_pretrained(dumpdir)
        aligner.tokenizer.save_pretrained(dumpdir)

    def linear(
        self,
        root_dir="/bigdata",
        data="opus",
        model="bert-base-multilingual-cased",
        name="linear-orth0.01",
    ):
        langs = "ar de es fr hi ru vi zh".split()
        _dir = f"{root_dir}/checkpoints/alignment/{data}"
        mapping = nn.ModuleDict()
        for lang in tqdm(langs):
            ckpt = f"{_dir}/en-{lang}-subset1/{model}-sim_linear/{name}/version_0/mapping.pth"
            mapping[lang] = torch.load(ckpt)
        os.makedirs(f"mapping/{name}", exist_ok=True)
        torch.save(mapping, f"mapping/{name}/{model}.pth")


if __name__ == "__main__":
    Fire(Main)
