import glob
from collections import defaultdict
from copy import deepcopy
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

import constant
from dataset.base import DUMMY_LABEL, LABEL_PAD_ID, Dataset
from enumeration import Split
from metric import convert_bio_to_spans


class TaggingDataset(Dataset):
    def before_load(self):
        self.max_len = min(self.max_len, self.tokenizer.max_len_single_sentence)
        self.shift = self.max_len // 2
        self.labels = self.get_labels()
        self.label2id = {label: idx for idx, label in enumerate(self.labels)}
        self.label2id[DUMMY_LABEL] = LABEL_PAD_ID

    @classmethod
    def nb_labels(cls):
        return len(cls.get_labels())

    @classmethod
    def get_labels(cls) -> List[str]:
        raise NotImplementedError

    def add_special_tokens(self, sent, labels):
        sent = self.tokenizer.build_inputs_with_special_tokens(sent)
        labels = self.tokenizer.build_inputs_with_special_tokens(labels)
        mask = self.tokenizer.get_special_tokens_mask(
            sent, already_has_special_tokens=True
        )
        sent, labels, mask = np.array(sent), np.array(labels), np.array(mask)
        label = labels * (1 - mask) + LABEL_PAD_ID * mask
        return sent, label

    def _process_example_helper(
        self, sent: List, labels: List
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:

        token_ids: List[int] = []
        label_ids: List[int] = []

        for token, label in zip(sent, labels):
            sub_tokens = self.tokenize(token)
            if not sub_tokens:
                continue
            sub_tokens = self.tokenizer.convert_tokens_to_ids(sub_tokens)

            if len(token_ids) + len(sub_tokens) >= self.max_len:
                # don't add more token
                yield self.add_special_tokens(token_ids, label_ids)

                token_ids = token_ids[-self.shift :]
                label_ids = [LABEL_PAD_ID] * len(token_ids)

            for i, sub_token in enumerate(sub_tokens):
                token_ids.append(sub_token)
                label_id = self.label2id[label] if i == 0 else LABEL_PAD_ID
                label_ids.append(label_id)

        yield self.add_special_tokens(token_ids, label_ids)

    def process_example(self, example: Dict) -> List[Dict]:
        sent: List = example["sent"]
        labels: List = example["labels"]

        data: List[Dict] = []
        if not sent:
            return data
        for src, tgt in self._process_example_helper(sent, labels):
            data.append({"sent": src, "labels": tgt, "lang": self.lang})
        return data


class ConllNER(TaggingDataset):
    @classmethod
    def get_labels(cls):
        return [
            "B-LOC",
            "B-MISC",
            "B-ORG",
            "B-PER",
            "I-LOC",
            "I-MISC",
            "I-ORG",
            "I-PER",
            "O",
        ]

    @classmethod
    def read_file(cls, filepath: str, lang: str, split: str) -> Iterator[Dict]:
        """Reads an empty line seperated data (word \t label)."""
        words: List[str] = []
        labels: List[str] = []
        with open(filepath, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    assert len(words) == len(labels)
                    yield {"sent": words, "labels": labels}
                    words, labels = [], []
                else:
                    word, label = line.split("\t")
                    words.append(word)
                    labels.append(label)
            if len(words) == len(labels) and words:
                yield {"sent": words, "labels": labels}

    @classmethod
    def get_file(cls, path: str, lang: str, split: str) -> Optional[str]:
        if split == Split.train:
            fp = f"{path}/{lang}/train.iob2.txt"
        elif split == Split.dev:
            fp = f"{path}/{lang}/dev.iob2.txt"
        elif split == Split.test:
            fp = f"{path}/{lang}/test.iob2.txt"
        else:
            raise ValueError(f"Unsupported split: {split}")
        return fp


class WikiAnnNER(TaggingDataset):
    @classmethod
    def get_labels(cls):
        return ["B-LOC", "B-ORG", "B-PER", "I-LOC", "I-ORG", "I-PER", "O"]

    @classmethod
    def read_file(cls, filepath: str, lang: str, split: str) -> Iterator[Dict]:
        """Reads an empty line seperated data (word \t label)."""
        words: List[str] = []
        labels: List[str] = []
        with open(filepath, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    assert len(words) == len(labels)
                    yield {"sent": words, "labels": labels}
                    words, labels = [], []
                else:
                    word, label = line.split("\t")
                    word = word.split(":", 1)[1]
                    words.append(word)
                    labels.append(label)
            if len(words) == len(labels) and words:
                yield {"sent": words, "labels": labels}

    @classmethod
    def write_example(cls, example: Dict, file_handler):
        assert "sent" in example
        assert "labels" in example
        assert len(example["sent"]) == len(example["labels"])
        for word, label in zip(example["sent"], example["labels"]):
            print(f"_:{word}\t{label}", file=file_handler)
        print("", file=file_handler)

    @classmethod
    def project_label(
        cls, example: Dict, translation: List[str], mapping: List[Tuple]
    ) -> Dict:
        # span projection
        src2tgt = defaultdict(list)
        for src_idx, tgt_idx in mapping:
            src2tgt[src_idx].append(tgt_idx)

        raw_labels = defaultdict(list)
        for entity, start, end in convert_bio_to_spans(example["labels"]):
            idx = set()
            for pos in range(start, end):
                for tgt_idx in src2tgt[pos]:
                    idx.add(tgt_idx)
            if not idx:  # no alignment
                continue
            tgt_start = min(idx)
            tgt_end = max(idx) + 1
            if (tgt_end - tgt_start) / len(idx) > 5:
                continue
            # new_span => (entity, tgt_start, tgt_end)
            for i, pos in enumerate(range(tgt_start, tgt_end)):
                if i == 0:
                    raw_labels[pos].append(f"B-{entity}")
                else:
                    raw_labels[pos].append(f"I-{entity}")

        words: List[str] = []
        labels: List[str] = []
        for i, word in enumerate(translation):
            raw_label = set(raw_labels[i])
            if not raw_label:
                silver_label = "O"
                # silver_label = DUMMY_LABEL
            elif len(raw_label) == 1:
                silver_label = list(raw_label)[0]
            else:
                begin, inside = set(), set()
                for label in raw_label:
                    if label.startswith("B"):
                        begin.add(label)
                    elif label.startswith("I"):
                        inside.add(label)
                if len(begin) > 0:
                    silver_label = list(begin)[0]
                elif len(inside) > 0:
                    silver_label = list(inside)[0]
                else:
                    raise ValueError("Impossible case")

            words.append(word)
            labels.append(silver_label)

        clean_labels: List[str] = deepcopy(labels)
        clean_spans = convert_bio_to_spans(labels)
        for entity, start, end in clean_spans:
            for i, pos in enumerate(range(start, end)):
                if i == 0:
                    clean_labels[pos] = f"B-{entity}"
                else:
                    clean_labels[pos] = f"I-{entity}"

        return {"sent": words, "labels": clean_labels}

    @classmethod
    def get_file(cls, path: str, lang: str, split: str) -> Optional[str]:
        if split == Split.train:
            fp = f"{path}/{lang}/train"
        elif split == Split.dev:
            fp = f"{path}/{lang}/dev"
        elif split == Split.test:
            fp = f"{path}/{lang}/test"
        else:
            raise ValueError("Unsupported split:", split)
        return fp


class UdPOS(TaggingDataset):
    @classmethod
    def get_labels(cls):
        return constant.UD_POS_LABELS

    @classmethod
    def read_file(cls, filepath: str, lang: str, split: str) -> Iterator[Dict]:
        words: List[str] = []
        labels: List[str] = []
        with open(filepath, "r") as f:
            for line in f.readlines():
                tok = line.strip().split("\t")
                if len(tok) < 2 or line[0] == "#":
                    assert len(words) == len(labels)
                    if words:
                        yield {"sent": words, "labels": labels}
                        words, labels = [], []
                if tok[0].isdigit():
                    word, label = tok[1], tok[3]
                    words.append(word)
                    labels.append(label)
            if len(words) == len(labels) and words:
                yield {"sent": words, "labels": labels}

    @classmethod
    def write_example(cls, example: Dict, file_handler):
        assert "sent" in example
        assert "labels" in example
        assert len(example["sent"]) == len(example["labels"])
        for idx, (word, label) in enumerate(
            zip(
                example["sent"],
                example["labels"],
            )
        ):
            fields = []
            fields.append(str(idx + 1))  # pos 0
            fields.append(word)  # pos 1
            fields.extend("_")  # pos 2
            fields.append(label)  # pos 3
            fields.extend(["_", "_", "_", "_", "_", "_"])  # pos 4-9
            print("\t".join(fields), file=file_handler)
        print("", file=file_handler)

    @classmethod
    def get_file(cls, path: str, lang: str, split: str) -> Optional[str]:
        if split == Split.train:
            fp = f"{path}/UD_{lang}/*-ud-train.conllu"
        elif split == Split.dev:
            fp = f"{path}/UD_{lang}/*-ud-dev.conllu"
        elif split == Split.test:
            fp = f"{path}/UD_{lang}/*-ud-test.conllu"
        else:
            raise ValueError(f"Unsupported split: {split}")
        _fp = glob.glob(fp)
        if len(_fp) == 1:
            return _fp[0]
        elif len(_fp) == 0:
            return None
        else:
            raise ValueError(f"Unsupported split: {split}")
