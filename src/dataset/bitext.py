from collections import Counter
from typing import Dict, Iterator, List, Optional

import numpy as np

from dataset.base import Dataset
from enumeration import Split


class Bitext(Dataset):
    def unpack_language(self, lang):
        src_lang, tgt_lang = lang.split("-")
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        return lang

    @classmethod
    def read_file(cls, filepath: str, lang: str, split: str) -> Iterator[Dict]:
        with open(f"{filepath}.text") as f_t, open(f"{filepath}.align") as f_a:
            for bitext, align in zip(f_t.readlines(), f_a.readlines()):
                src, tgt = bitext.split(" ||| ")
                src_sent, tgt_sent = src.split(), tgt.split()
                align_idx = [tuple(map(int, a.split("-"))) for a in align.split()]
                yield {"src_sent": src_sent, "tgt_sent": tgt_sent, "align": align_idx}

    def _process_sent(self, sent: str):
        word_pos = dict()
        tokens = [self.tokenizer.cls_token]
        for i, token in enumerate(sent):
            sub_tokens = self.tokenizer.tokenize(token)
            if len(tokens) + len(sub_tokens) >= self.max_len:
                break
            if sub_tokens:
                word_pos[i] = len(tokens)
                tokens.extend(sub_tokens)
        tokens.append(self.tokenizer.sep_token)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return np.array(token_ids), word_pos

    def _process_align(self, src, trg, align):
        # check same token
        align = [(sa, ta) for sa, ta in align if src[sa] != trg[ta]]
        # one-to-one
        scnt = Counter([sa for sa, ta in align])
        tcnt = Counter([ta for sa, ta in align])
        align = [(sa, ta) for sa, ta in align if scnt[sa] == 1 and tcnt[ta] == 1]
        return align

    def process_example(self, example: Dict) -> List[Dict]:
        src_sent, src_word_pos = self._process_sent(example["src_sent"])
        tgt_sent, tgt_word_pos = self._process_sent(example["tgt_sent"])
        if len(src_sent) == 0 or len(tgt_sent) == 0:
            return []
        align = self._process_align(
            example["src_sent"], example["tgt_sent"], example["align"]
        )
        src_align, tgt_align = [], []  # align cls_token
        for sa, ta in align:
            if sa in src_word_pos and ta in tgt_word_pos:
                if src_word_pos[sa] < self.max_len and tgt_word_pos[ta] < self.max_len:
                    src_align.append(src_word_pos[sa])
                    tgt_align.append(tgt_word_pos[ta])
        if len(src_align) == 0:
            return []
        src_align, tgt_align = np.array(src_align), np.array(tgt_align)
        return [
            {
                "src_sent": src_sent,
                "tgt_sent": tgt_sent,
                "src_align": src_align,
                "tgt_align": tgt_align,
                "src_lang": self.src_lang,
                "tgt_lang": self.tgt_lang,
                "lang": self.lang,
            }
        ]

    @classmethod
    def get_file(cls, path: str, lang: str, split: str) -> Optional[str]:
        if split == Split.train:
            fp = f"{path}/{lang}.trn"
        elif split == Split.dev:
            fp = f"{path}/{lang}.val"
        elif split == Split.test:
            fp = f"{path}/{lang}.tst"
        else:
            raise ValueError("Unsupported split:", split)
        return fp
