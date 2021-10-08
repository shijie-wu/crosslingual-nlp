import glob
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

import constant
from dataset.base import DUMMY_LABEL, LABEL_PAD_ID, Dataset
from enumeration import Split


class ParsingDataset(Dataset):
    def __init__(
        self,
        *,
        max_len_unit: str,
        **kwargs,
    ):
        assert max_len_unit in ["word", "subword"]
        self.max_len_unit = max_len_unit
        super().__init__(**kwargs)

    def before_load(self):
        self.max_len = min(self.max_len, self.tokenizer.max_len_single_sentence)
        self.labels = self.get_labels()
        self.label2id = {label: idx for idx, label in enumerate(self.labels)}
        self.label2id[DUMMY_LABEL] = LABEL_PAD_ID
        self.pos_tags = self.get_pos_tags()
        self.pos2id = {pos: idx for idx, pos in enumerate(self.pos_tags)}
        self.pos2id[DUMMY_LABEL] = LABEL_PAD_ID

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

    @classmethod
    def read_file(cls, filepath: str, lang: str, split: str) -> Iterator[Dict]:
        with open(filepath, "r") as f:
            sent: List[str] = []
            pos_tags: List[str] = []
            heads: List[int] = []
            labels: List[str] = []
            for line in f.readlines():
                tok = line.strip().split("\t")
                if len(tok) < 2 or line[0] == "#":
                    if sent:
                        yield {
                            "sent": sent,
                            "pos_tags": pos_tags,
                            "heads": heads,
                            "labels": labels,
                        }
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
                yield {
                    "sent": sent,
                    "pos_tags": pos_tags,
                    "heads": heads,
                    "labels": labels,
                }

    @classmethod
    def write_example(cls, example: Dict, file_handler):
        assert "sent" in example
        assert "pos_tags" in example
        assert "heads" in example
        assert "labels" in example
        assert len(example["sent"]) == len(example["pos_tags"])
        assert len(example["sent"]) == len(example["heads"])
        assert len(example["sent"]) == len(example["labels"])
        for idx, (word, pos, head, label) in enumerate(
            zip(
                example["sent"],
                example["pos_tags"],
                example["heads"],
                example["labels"],
            )
        ):
            fields = []
            fields.append(str(idx + 1))  # pos 0
            fields.append(word)  # pos 1
            fields.extend("_")  # pos 2
            fields.append(pos)  # pos 3
            fields.extend(["_", "_"])  # pos 4,5
            fields.append(str(head))  # pos 6
            fields.append(f"{label}:_")  # pos 7
            fields.extend(["_", "_"])  # pos 8,9
            print("\t".join(fields), file=file_handler)
        print("", file=file_handler)

    @classmethod
    def project_label(
        cls, example: Dict, translation: List[str], mapping: List[Tuple]
    ) -> Dict:
        src2tgt = defaultdict(list)
        for src_idx, tgt_idx in mapping:
            src2tgt[src_idx].append(tgt_idx)

        child2parent = dict()
        for i, head in enumerate(example["heads"]):
            # the head index starts from 1
            child2parent[i] = head - 1
        distance2root = dict()
        for i in child2parent.keys():
            dist, pos = 0, i
            while child2parent[pos] != -1:
                dist += 1
                pos = child2parent[pos]
            distance2root[i] = dist

        # token projection
        raw_labels = defaultdict(list)
        for src_idx, tgt_idx in mapping:
            # src = example["sent"][src_idx]
            pos_tag = example["pos_tags"][src_idx]
            head = example["heads"][src_idx]
            label = example["labels"][src_idx]
            # tgt = translation[tgt_idx]
            # the head index starts from 1
            # while the mapping index starts from 0
            # hence `head - 1` when indexing src2tgt
            raw_labels[tgt_idx].append(
                (src_idx, pos_tag, head, src2tgt[head - 1], label)
            )

        sent: List[str] = []
        pos_tags: List[str] = []
        heads: List[int] = []
        labels: List[str] = []
        for i, word in enumerate(translation):
            if not raw_labels[i]:  # no alignment at all
                sent.append(word)
                pos_tags.append(DUMMY_LABEL)
                heads.append(LABEL_PAD_ID)
                labels.append(DUMMY_LABEL)
                continue

            # used to pick the first one
            # break tie by selecting the node closest to root
            raw_label = min(raw_labels[i], key=lambda x: distance2root[x[0]])

            src_idx, slvr_pos, src_head, _slvr_head, slvr_label = raw_label

            if not _slvr_head:  # no silver head
                parent = child2parent[src_idx]
                assert not src2tgt[parent] and parent == src_head - 1
                while parent != -1:
                    parent = child2parent[parent]
                    # we pick ancestor's alignment as head
                    if src2tgt[parent]:
                        _slvr_head = src2tgt[parent]
                        break

            if _slvr_head:
                # break tie by selecting the right most node
                # the head index starts from 1 => + 1
                slvr_head = max(_slvr_head) + 1
            else:  # no ancestor has alignment
                slvr_head = 0

            sent.append(word)
            pos_tags.append(slvr_pos)
            heads.append(slvr_head)
            labels.append(slvr_label)

        return {
            "sent": sent,
            "pos_tags": pos_tags,
            "heads": heads,
            "labels": labels,
        }

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

    def process_example(self, example: Dict) -> List[Dict]:
        sent: List = example["sent"]
        pos_tags: List = example["pos_tags"]
        heads: List = example["heads"]
        labels: List = example["labels"]

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

                if head >= 0:
                    head_id = word2subword_pos[head] if i == 0 else -1
                else:
                    assert head == -1
                    assert pos_tag == DUMMY_LABEL
                    assert label == DUMMY_LABEL
                    head_id = -1
                head_ids.append(head_id)

                label_id = self.label2id[label] if i == 0 else LABEL_PAD_ID
                label_ids.append(label_id)

        token_ids, pos_ids, head_ids, label_ids = self.add_special_tokens(
            token_ids, pos_ids, head_ids, label_ids
        )
        return [
            {
                "sent": token_ids,
                "pos_tags": pos_ids,
                "heads": head_ids,
                "labels": label_ids,
                "lang": self.lang,
            }
        ]

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
