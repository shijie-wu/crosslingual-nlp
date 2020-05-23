import glob
from typing import Dict, Iterator, List, Optional, Tuple

import langcodes
import numpy as np
import torch
import transformers
from nltk.tokenize import sent_tokenize as nltk_sent_tokenize
from tqdm import tqdm

import constant
from enumeration import Split


def sent_tokenize(text, lang="en"):
    lang = langcodes.Language(lang).language_name().lower()
    try:
        return nltk_sent_tokenize(text, language=lang)
    except LookupError:
        return nltk_sent_tokenize(text)


class Tokenizer(transformers.PreTrainedTokenizer):
    pass


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        tokenizer: Tokenizer,
        filepath: str,
        lang: str,
        split: Optional[Split] = None,
        max_len: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.filepath = filepath
        self.lang = self.unpack_language(lang)
        self.split = split
        if max_len is not None:
            assert 0 < max_len <= self.tokenizer.max_len
        self.max_len = max_len if max_len is not None else self.tokenizer.max_len
        self.data: List[Dict[str, np.ndarray]] = []

        self.before_read_file()
        self.read_file(filepath)

    def unpack_language(self, lang):
        return lang

    def tokenize(self, token):
        if isinstance(self.tokenizer, transformers.XLMTokenizer):
            sub_words = self.tokenizer.tokenize(token, lang=self.lang)
        else:
            sub_words = self.tokenizer.tokenize(token)
        if isinstance(self.tokenizer, transformers.XLMRobertaTokenizer):
            if sub_words[0] == "▁":
                sub_words = sub_words[1:]
        return sub_words

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def before_read_file(self):
        pass

    def read_file(self, filepath: str):
        raise NotImplementedError

    @classmethod
    def get_file(cls, path: str, lang: str, split: Split) -> Optional[str]:
        raise NotImplementedError


class ClassificationDataset(Dataset):
    def before_read_file(self):
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

    def _process_file(self, filepath) -> Iterator[Tuple[str, Optional[str], str]]:
        raise NotImplementedError

    def process_example(
        self, sent1: str, sent2: Optional[str], label: str
    ) -> Optional[Dict]:
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
        return {"sent": sent, "segment": segment, "label": label, "lang": self.lang}

    def read_file(self, filepath: str):
        assert self.data == []
        data = []
        for sent1, sent2, label in tqdm(self._process_file(filepath), desc="read data"):
            d = self.process_example(sent1, sent2, label)
            if d is not None:
                data.append(d)
        self.data = data


class Xnli(ClassificationDataset):
    @classmethod
    def get_labels(cls) -> List[str]:
        return ["contradiction", "entailment", "neutral"]

    def _process_file(self, filepath) -> Iterator[Tuple[str, Optional[str], str]]:
        for row in self.read_csv(filepath, delimiter="\t"):
            if self.split == Split.train:
                sent1 = row["premise"]
                sent2 = row["hypo"]
                label = row["label"]
                if label == "contradictory":
                    label = "contradiction"
                yield sent1, sent2, label
            elif self.split == Split.dev or self.split == Split.test:
                if row["language"] != self.lang:
                    continue
                sent1 = row["sentence1"]
                sent2 = row["sentence2"]
                label = row["gold_label"]
                yield sent1, sent2, label
            else:
                raise ValueError(f"Unsupported split: {self.split}")

    @classmethod
    def get_file(cls, path: str, lang: str, split: Split) -> Optional[str]:
        if split == Split.train:
            fp = f"{path}/multinli/multinli.train.{lang}.tsv"
        elif split == Split.dev:
            fp = f"{path}/xnli.dev.tsv"
        elif split == Split.test:
            fp = f"{path}/xnli.test.tsv"
        else:
            raise ValueError(f"Unsupported split: {split}")
        return fp


class MLDoc(ClassificationDataset):
    @classmethod
    def get_labels(cls) -> List[str]:
        return ["CCAT", "ECAT", "GCAT", "MCAT"]

    def _process_file(self, filepath) -> Iterator[Tuple[str, Optional[str], str]]:
        with open(filepath, "r") as fp:
            for line in fp.readlines():
                line = line.strip()
                label, sents = line.strip().split("\t", maxsplit=1)

                # sent1, sent2 = sents, None
                sents = sent_tokenize(sents, lang=self.lang)
                sent1 = sents[0]

                sent2: Optional[str] = None
                if len(sents) > 1:
                    sent2 = sents[1]

                yield sent1, sent2, label

    @classmethod
    def get_file(cls, path: str, lang: str, split: Split) -> Optional[str]:
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
    def get_file(cls, path: str, lang: str, split: Split) -> Optional[str]:
        if split == Split.train:
            fp = f"{path}/langid.train"
        elif split == Split.dev:
            fp = f"{path}/langid.dev"
        elif split == Split.test:
            fp = f"{path}/langid.test"
        else:
            raise ValueError(f"Unsupported split: {split}")
        return fp


LABEL_PAD_ID = -1


class TaggingDataset(Dataset):
    def before_read_file(self):
        self.max_len = min(self.max_len, self.tokenizer.max_len_single_sentence)
        self.shift = self.max_len // 2
        self.labels = self.get_labels()
        self.label2id = {label: idx for idx, label in enumerate(self.labels)}

    @classmethod
    def nb_labels(cls):
        return len(cls.get_labels())

    @classmethod
    def get_labels(cls) -> List[str]:
        raise NotImplementedError

    def _process_file(self, filepath) -> Iterator[Tuple[List, List]]:
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

    def process_example(self, sent: List, labels: List) -> Optional[List[Dict]]:
        data = []
        if not sent:
            return None
        for src, tgt in self._process_example_helper(sent, labels):
            data.append({"sent": src, "labels": tgt, "lang": self.lang})
        return data

    def read_file(self, filepath: str):
        assert self.data == []
        data = []
        for sent, labels in tqdm(self._process_file(filepath), desc="read data"):
            d = self.process_example(sent, labels)
            if d is not None:
                data.extend(d)
        self.data = data


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

    def _process_file(self, filepath: str):
        """Reads an empty line seperated data (word \t label)."""
        words: List[str] = []
        labels: List[str] = []
        with open(filepath, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    assert len(words) == len(labels)
                    yield words, labels
                    words, labels = [], []
                else:
                    word, label = line.split("\t")
                    words.append(word)
                    labels.append(label)
            if len(words) == len(labels) and words:
                yield words, labels

    @classmethod
    def get_file(cls, path: str, lang: str, split: Split) -> Optional[str]:
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

    def _process_file(self, filepath: str):
        """Reads an empty line seperated data (word \t label)."""
        words: List[str] = []
        labels: List[str] = []
        with open(filepath, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    assert len(words) == len(labels)
                    yield words, labels
                    words, labels = [], []
                else:
                    word, label = line.split("\t")
                    word = word.split(":", 1)[1]
                    words.append(word)
                    labels.append(label)
            if len(words) == len(labels) and words:
                yield words, labels

    @classmethod
    def get_file(cls, path: str, lang: str, split: Split) -> Optional[str]:
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

    def _process_file(self, filepath: str):
        words: List[str] = []
        labels: List[str] = []
        with open(filepath, "r") as f:
            for line in f.readlines():
                tok = line.strip().split("\t")
                if len(tok) < 2 or line[0] == "#":
                    assert len(words) == len(labels)
                    if words:
                        yield words, labels
                        words, labels = [], []
                if tok[0].isdigit():
                    word, label = tok[1], tok[3]
                    words.append(word)
                    labels.append(label)
            if len(words) == len(labels) and words:
                yield words, labels

    @classmethod
    def get_file(cls, path: str, lang: str, split: Split) -> Optional[str]:
        if split == Split.train:
            fp = f"{path}/UD_{lang}/*-ud-train.conllu"
        elif split == Split.dev:
            fp = f"{path}/UD_{lang}/*-ud-dev.conllu"
        elif split == Split.test:
            fp = f"{path}/UD_{lang}/*-ud-test.conllu"
        else:
            raise ValueError(f"Unsupported split: {split}")
        fp = glob.glob(fp)
        if len(fp) == 1:
            return fp[0]
        elif len(fp) == 0:
            return None
        else:
            raise ValueError(f"Unsupported split: {split}")


class ParsingDataset(Dataset):
    def __init__(
        self, *, max_len_unit: str, **kwargs,
    ):
        assert max_len_unit in ["word", "subword"]
        self.max_len_unit = max_len_unit
        super().__init__(**kwargs)

    def before_read_file(self):
        self.max_len = min(self.max_len, self.tokenizer.max_len_single_sentence)
        self.labels = self.get_labels()
        self.label2id = {label: idx for idx, label in enumerate(self.labels)}
        self.pos_tags = self.get_pos_tags()
        self.pos2id = {pos: idx for idx, pos in enumerate(self.pos_tags)}

    @classmethod
    def nb_labels(cls):
        return len(cls.get_labels())

    @classmethod
    def get_labels(cls) -> List[str]:
        return constant.UD_HEAD_LABELS

    @classmethod
    def nb_pos_tags(cls):
        return len(cls.get_pos_tags())

    @classmethod
    def get_pos_tags(cls) -> List[str]:
        return constant.UD_POS_LABELS

    def _process_file(self, filepath) -> Iterator[Tuple[List, List, List, List]]:
        with open(filepath, "r") as f:
            sent: List[str] = []
            pos_tags: List[str] = []
            heads: List[int] = []
            labels: List[str] = []
            for line in f.readlines():
                tok = line.strip().split("\t")
                if len(tok) < 2 or line[0] == "#":
                    if sent:
                        yield sent, pos_tags, heads, labels
                        sent = []
                        pos_tags = []
                        heads = []
                        labels = []
                if tok[0].isdigit():
                    word, pos, head, label = tok[1], tok[3], tok[6], tok[7]
                    sent.append(word)
                    pos_tags.append(pos)
                    heads.append(int(head))
                    labels.append(label.split(":")[0])
            if sent:
                yield sent, pos_tags, heads, labels

    def add_special_tokens(self, sent, pos_tags, heads, labels):
        sent = self.tokenizer.build_inputs_with_special_tokens(sent)
        pos_tags = self.tokenizer.build_inputs_with_special_tokens(pos_tags)
        heads = self.tokenizer.build_inputs_with_special_tokens(heads)
        labels = self.tokenizer.build_inputs_with_special_tokens(labels)
        mask = self.tokenizer.get_special_tokens_mask(
            sent, already_has_special_tokens=True
        )
        sent = np.array(sent)
        pos_tags = np.array(pos_tags)
        heads = np.array(heads)
        labels = np.array(labels)
        mask = np.array(mask)
        pos_tags = pos_tags * (1 - mask) + LABEL_PAD_ID * mask
        heads = heads * (1 - mask) + -1 * mask
        labels = labels * (1 - mask) + LABEL_PAD_ID * mask
        return sent, pos_tags, heads, labels

    def process_example(
        self, sent: List, pos_tags: List, heads: List, labels: List
    ) -> Optional[Dict]:

        token_ids: List[int] = []
        pos_ids: List[int] = []
        head_ids: List[int] = []
        label_ids: List[int] = []

        tokens = [self.tokenize(w) for w in sent]
        # We use this to convert head position in UD to the first subword position in the
        # tokenized sentence. As UD head position is 1 based, we assume the model prepend
        # *one* speical token to a sentence.
        word2subword_pos = np.cumsum([0, 1] + [len(w) for w in tokens])

        # **max_len in parsing is the number of word instead of subword**
        if self.max_len_unit == "word":
            tokens = tokens[: self.max_len]
            max_len = self.tokenizer.max_len_single_sentence
        else:
            max_len = self.max_len
        for sub_tokens, pos_tag, head, label in zip(tokens, pos_tags, heads, labels):
            sub_tokens = self.tokenizer.convert_tokens_to_ids(sub_tokens)

            if len(token_ids) + len(sub_tokens) >= max_len:
                # don't add more token
                break

            for i, sub_token in enumerate(sub_tokens):
                token_ids.append(sub_token)

                pos_id = self.pos2id[pos_tag] if i == 0 else LABEL_PAD_ID
                pos_ids.append(pos_id)

                assert head >= 0
                head_id = word2subword_pos[head] if i == 0 else -1
                head_ids.append(head_id)

                label_id = self.label2id[label] if i == 0 else LABEL_PAD_ID
                label_ids.append(label_id)

        token_ids, pos_ids, head_ids, label_ids = self.add_special_tokens(
            token_ids, pos_ids, head_ids, label_ids
        )
        return {
            "sent": token_ids,
            "pos_tags": pos_ids,
            "heads": head_ids,
            "labels": label_ids,
            "lang": self.lang,
        }

    def read_file(self, filepath: str):
        assert self.data == []
        data = []
        for sent, pos_tags, heads, labels in tqdm(
            self._process_file(filepath), desc="read data"
        ):
            d = self.process_example(sent, pos_tags, heads, labels)
            if d is not None:
                data.append(d)
        self.data = data

    @classmethod
    def get_file(cls, path: str, lang: str, split: Split) -> Optional[str]:
        if split == Split.train:
            fp = f"{path}/UD_{lang}/*-ud-train.conllu"
        elif split == Split.dev:
            fp = f"{path}/UD_{lang}/*-ud-dev.conllu"
        elif split == Split.test:
            fp = f"{path}/UD_{lang}/*-ud-test.conllu"
        else:
            raise ValueError(f"Unsupported split: {split}")
        fp = glob.glob(fp)
        if len(fp) == 1:
            return fp[0]
        elif len(fp) == 0:
            return None
        else:
            raise ValueError(f"Unsupported split: {split}")
