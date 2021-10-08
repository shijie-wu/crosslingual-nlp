from functools import partial
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import transformers
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

tqdm.monitor_interval = 0
tqdm = partial(tqdm, bar_format="{l_bar}{r_bar}")


LABEL_PAD_ID = -1
DUMMY_LABEL = "DUMMY_LABEL"


class Tokenizer(transformers.PreTrainedTokenizer):
    pass


class Dataset(TorchDataset):
    def __init__(
        self,
        *,
        tokenizer: Tokenizer,
        filepath: str,
        lang: str,
        split: Optional[str] = None,
        max_len: Optional[int] = None,
        subset_ratio: float = 1,
        subset_count: int = -1,
        subset_seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.filepath = filepath
        self.lang = self.unpack_language(lang)
        self.split = split
        if max_len is not None:
            assert 0 < max_len <= self.tokenizer.max_len_single_sentence
        self.max_len = (
            max_len if max_len is not None else self.tokenizer.max_len_single_sentence
        )
        self.data: List[Dict[str, np.ndarray]] = []

        assert 0 < subset_ratio <= 1
        assert not (
            subset_ratio < 1 and subset_count > 0
        ), "subset_ratio and subset_count is mutally exclusive"
        self.subset_ratio = subset_ratio
        self.subset_count = subset_count
        self.subset_seed = subset_seed

        self.before_load()
        self.load()

    def unpack_language(self, lang):
        return lang

    def tokenize(self, token):
        if isinstance(self.tokenizer, transformers.XLMTokenizer):
            sub_words = self.tokenizer.tokenize(token, lang=self.lang)
        else:
            sub_words = self.tokenizer.tokenize(token)
        if isinstance(self.tokenizer, transformers.XLMRobertaTokenizer):
            if not sub_words:
                return []
            if sub_words[0] == "▁":
                sub_words = sub_words[1:]
        return sub_words

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def before_load(self):
        pass

    def load(self):
        assert self.data == []

        examples = []
        for ex in tqdm(
            self.read_file(self.filepath, self.lang, self.split), desc="read data"
        ):
            examples.append(ex)

        if self.subset_count > 0 or self.subset_ratio < 1:
            if self.subset_count > 0:
                subset_size = self.subset_count
            elif self.subset_ratio < 1:
                subset_size = int(len(examples) * self.subset_ratio)
            else:
                raise ValueError("subset_ratio and subset_count is mutally exclusive")

            print(
                f"taking {subset_size} subset (total {len(examples)}) from {self.filepath}"
            )

            seed = np.random.RandomState(self.subset_seed)
            examples = seed.permutation(examples)[:subset_size]

        data = []
        for example in tqdm(examples, desc="parse data"):
            data.extend(self.process_example(example))
        self.data = data

    @classmethod
    def get_file(cls, path: str, lang: str, split: str) -> Optional[str]:
        raise NotImplementedError

    @classmethod
    def read_file(cls, filepath: str, lang: str, split: str) -> Iterator[Dict]:
        raise NotImplementedError

    def process_example(self, example: Dict) -> List[Dict]:
        raise NotImplementedError

    @classmethod
    def write_example(cls, example: Dict, file_handler):
        raise NotImplementedError

    @classmethod
    def project_label(
        cls, example: Dict, translation: List[str], mapping: List[Tuple]
    ) -> Dict:
        raise NotImplementedError
