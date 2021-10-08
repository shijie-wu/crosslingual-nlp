from typing import Dict, Iterator, List, Optional

import langcodes
import numpy as np
from nltk.tokenize import sent_tokenize as nltk_sent_tokenize

import constant
from dataset.base import Dataset
from enumeration import Split


def sent_tokenize(text, lang="en"):
    lang = langcodes.Language(lang).language_name().lower()
    try:
        return nltk_sent_tokenize(text, language=lang)
    except (LookupError, KeyError):
        return nltk_sent_tokenize(text)


class ClassificationDataset(Dataset):
    def before_load(self):
        self.max_len = min(self.max_len, self.tokenizer.max_len_sentences_pair)
        self.labels = self.get_labels()
        self.label2id = {label: idx for idx, label in enumerate(self.labels)}

    @classmethod
    def nb_labels(cls) -> int:
        return len(cls.get_labels())

    @classmethod
    def get_labels(cls) -> List[str]:
        raise NotImplementedError

    @classmethod
    def read_csv(cls, filename, delimiter) -> Iterator[Dict]:
        with open(filename, "r") as fp:
            keys = fp.readline().strip().split(delimiter)
            for line in fp.readlines():
                vals = line.strip().split(delimiter)
                assert len(keys) == len(vals)
                yield {k: v for k, v in zip(keys, vals)}

    def process_example(self, example: Dict) -> List[Dict]:
        sent1: str = example["sent1"]
        sent2: Optional[str] = example["sent2"]
        label: str = example["label"]

        tknzr = self.tokenizer

        tokens1 = self.tokenize(sent1)
        tokens1 = tknzr.convert_tokens_to_ids(tokens1)
        num_tokens = len(tokens1)

        tokens2 = None
        if sent2 is not None:
            tokens2 = self.tokenize(sent2)
            tokens2 = tknzr.convert_tokens_to_ids(tokens2)
            num_tokens += len(tokens2)

        tokens1, tokens2, _ = tknzr.truncate_sequences(
            tokens1, tokens2, num_tokens_to_remove=num_tokens - self.max_len
        )
        sent = tknzr.build_inputs_with_special_tokens(tokens1, tokens2)
        segment = tknzr.create_token_type_ids_from_sequences(tokens1, tokens2)
        assert len(sent) == len(segment)
        sent, segment = np.array(sent), np.array(segment)

        label = np.array(self.label2id[label]).reshape(-1)
        return [{"sent": sent, "segment": segment, "label": label, "lang": self.lang}]


class Xnli(ClassificationDataset):
    @classmethod
    def get_labels(cls) -> List[str]:
        return ["contradiction", "entailment", "neutral"]

    @classmethod
    def read_file(cls, filepath: str, lang: str, split: str) -> Iterator[Dict]:
        for row in cls.read_csv(filepath, delimiter="\t"):
            if split == Split.train:
                sent1 = row["premise"]
                sent2 = row["hypo"]
                label = row["label"]
                if label == "contradictory":
                    label = "contradiction"
                yield {"sent1": sent1, "sent2": sent2, "label": label}
            elif split == Split.dev or split == Split.test:
                if row["language"] != lang:
                    continue
                sent1 = row["sentence1"]
                sent2 = row["sentence2"]
                label = row["gold_label"]
                yield {"sent1": sent1, "sent2": sent2, "label": label}
            else:
                raise ValueError(f"Unsupported split: {split}")

    @classmethod
    def get_file(cls, path: str, lang: str, split: str) -> Optional[str]:
        if split == Split.train:
            fp = f"{path}/multinli/multinli.train.{lang}.tsv"
        elif split == Split.dev:
            fp = f"{path}/xnli.dev.tsv"
        elif split == Split.test:
            fp = f"{path}/xnli.test.tsv"
        else:
            raise ValueError(f"Unsupported split: {split}")
        return fp


class PawsX(ClassificationDataset):
    @classmethod
    def get_labels(cls) -> List[str]:
        return ["0", "1"]

    @classmethod
    def read_file(cls, filepath: str, lang: str, split: str) -> Iterator[Dict]:
        for row in cls.read_csv(filepath, delimiter="\t"):
            sent1 = row["sentence1"]
            sent2 = row["sentence2"]
            label = row["label"]
            yield {"sent1": sent1, "sent2": sent2, "label": label}

    @classmethod
    def get_file(cls, path: str, lang: str, split: str) -> Optional[str]:
        if split == Split.train:
            if lang == "en":
                fp = f"{path}/{lang}/train.tsv"
            else:
                fp = f"{path}/{lang}/translated_train.tsv"
        elif split == Split.dev:
            fp = f"{path}/{lang}/dev_2k.tsv"
        elif split == Split.test:
            fp = f"{path}/{lang}/test_2k.tsv"
        else:
            raise ValueError(f"Unsupported split: {split}")
        return fp


class MLDoc(ClassificationDataset):
    @classmethod
    def get_labels(cls) -> List[str]:
        return ["CCAT", "ECAT", "GCAT", "MCAT"]

    @classmethod
    def read_file(cls, filepath: str, lang: str, split: str) -> Iterator[Dict]:
        with open(filepath, "r") as fp:
            for line in fp.readlines():
                line = line.strip()
                label, sents = line.strip().split("\t", maxsplit=1)

                # sent1, sent2 = sents, None
                sents = sent_tokenize(sents, lang=lang)
                sent1 = sents[0]

                sent2: Optional[str] = None
                if len(sents) > 1:
                    sent2 = sents[1]

                yield {"sent1": sent1, "sent2": sent2, "label": label}

    @classmethod
    def get_file(cls, path: str, lang: str, split: str) -> Optional[str]:
        lang = {
            "zh": "chinese",
            "en": "english",
            "fr": "french",
            "de": "german",
            "it": "italian",
            "ja": "japanese",
            "ru": "russian",
            "es": "spanish",
        }[lang]
        if split == Split.train:
            fp = f"{path}/{lang}.train.1000"
        elif split == Split.dev:
            fp = f"{path}/{lang}.dev"
        elif split == Split.test:
            fp = f"{path}/{lang}.test"
        else:
            raise ValueError(f"Unsupported split: {split}")
        return fp


class LanguageID(MLDoc):
    @classmethod
    def get_labels(cls) -> List[str]:
        return constant.LANDID_LABEL

    @classmethod
    def get_file(cls, path: str, lang: str, split: str) -> Optional[str]:
        if split == Split.train:
            fp = f"{path}/langid.train"
        elif split == Split.dev:
            fp = f"{path}/langid.dev"
        elif split == Split.test:
            fp = f"{path}/langid.test"
        else:
            raise ValueError(f"Unsupported split: {split}")
        return fp
