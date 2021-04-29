import hashlib
import json
import os
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from copy import deepcopy
from functools import partial
from typing import Dict, List, Optional, Type

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler
from transformers import AutoConfig, AutoModel, AutoTokenizer

import constant
import module
import util
from dataset import Dataset
from enumeration import Schedule, Split, Task
from metric import Metric


class Model(pl.LightningModule):
    def __init__(self, hparams):
        super(Model, self).__init__()
        self.optimizer = None
        self.scheduler = None
        self._metric: Optional[Metric] = None
        self.metrics: Dict[str, Metric] = dict()
        self.trn_datasets: List[Dataset] = None
        self.val_datasets: List[Dataset] = None
        self.tst_datasets: List[Dataset] = None
        self.padding: Dict[str, int] = {}
        self.base_dir: str = ""

        self._batch_per_epoch: int = -1
        self._comparsion: Optional[str] = None
        self._selection_criterion: Optional[str] = None

        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        # self.hparams: Namespace = hparams
        self.save_hyperparameters(hparams)
        pl.seed_everything(hparams.seed)

        self.tokenizer = AutoTokenizer.from_pretrained(hparams.pretrain)
        self.model = self.build_model()
        self.freeze_layers()

        self.weight = nn.Parameter(torch.zeros(self.num_layers))
        self.mapping = None
        if hparams.mapping:
            assert os.path.isfile(hparams.mapping)
            self.mapping = torch.load(hparams.mapping)
            util.freeze(self.mapping)

        self.projector = self.build_projector()
        self.dropout = module.InputVariationalDropout(hparams.input_dropout)

    def build_model(self):
        config = AutoConfig.from_pretrained(
            self.hparams.pretrain, output_hidden_states=True
        )
        model = AutoModel.from_pretrained(self.hparams.pretrain, config=config)
        return model

    def freeze_layers(self):
        if self.hparams.freeze_layer == -1:
            return
        elif self.hparams.freeze_layer >= 0:
            for i in range(self.hparams.freeze_layer + 1):
                if i == 0:
                    print("freeze embeddings")
                    self.freeze_embeddings()
                else:
                    print(f"freeze layer {i}")
                    self.freeze_layer(i)

    def freeze_embeddings(self):
        if isinstance(self.model, transformers.BertModel) or isinstance(
            self.model, transformers.RobertaModel
        ):
            util.freeze(self.model.embeddings)
        elif isinstance(self.model, transformers.XLMModel):
            util.freeze(self.model.position_embeddings)
            if self.model.n_langs > 1 and self.model.use_lang_emb:
                util.freeze(self.model.lang_embeddings)
            util.freeze(self.model.embeddings)
        else:
            raise ValueError("Unsupported model")

    def freeze_layer(self, layer):
        if isinstance(self.model, transformers.BertModel) or isinstance(
            self.model, transformers.RobertaModel
        ):
            util.freeze(self.model.encoder.layer[layer - 1])
        elif isinstance(self.model, transformers.XLMModel):
            util.freeze(self.model.attentions[layer - 1])
            util.freeze(self.model.layer_norm1[layer - 1])
            util.freeze(self.model.ffns[layer - 1])
            util.freeze(self.model.layer_norm2[layer - 1])
        else:
            raise ValueError("Unsupported model")

    @property
    def hidden_size(self):
        if isinstance(self.model, transformers.BertModel) or isinstance(
            self.model, transformers.RobertaModel
        ):
            return self.model.config.hidden_size
        elif isinstance(self.model, transformers.XLMModel):
            return self.model.dim
        else:
            raise ValueError("Unsupported model")

    @property
    def num_layers(self):
        if isinstance(self.model, transformers.BertModel) or isinstance(
            self.model, transformers.RobertaModel
        ):
            return self.model.config.num_hidden_layers + 1
        elif isinstance(self.model, transformers.XLMModel):
            return self.model.n_layers + 1
        else:
            raise ValueError("Unsupported model")

    @property
    def batch_per_epoch(self):
        if self.trn_datasets is None:
            self.trn_datasets = self.prepare_datasets(Split.train)

        if self._batch_per_epoch < 0:
            total_datasize = sum([len(d) for d in self.trn_datasets])
            self._batch_per_epoch = np.ceil(total_datasize / self.hparams.batch_size)

        return self._batch_per_epoch

    @property
    def selection_criterion(self):
        assert self._selection_criterion is not None
        return self._selection_criterion

    @property
    def comparsion(self):
        assert self._comparsion is not None
        return self._comparsion

    def setup_metrics(self):
        assert self._metric is not None
        langs = self.hparams.trn_langs + self.hparams.val_langs + self.hparams.tst_langs
        langs = sorted(list(set(langs)))
        for lang in langs:
            self.metrics[lang] = deepcopy(self._metric)
        self.reset_metrics()

    def reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()

    def build_projector(self):
        hparams = self.hparams
        if hparams.projector == "id":
            return module.Identity()
        elif hparams.projector == "meanpool":
            return module.MeanPooling()
        elif hparams.projector == "transformer":
            return module.Transformer(
                input_dim=self.hidden_size,
                hidden_dim=hparams.projector_trm_hidden_size,
                num_heads=hparams.projector_trm_num_heads,
                dropout=hparams.projector_dropout,
                num_layers=hparams.projector_trm_num_layers,
            )
        else:
            raise ValueError(hparams.projector)

    def get_mask(self, sent: Tensor):
        mask = (sent != self.tokenizer.pad_token_id).long()
        return mask

    def encode_sent(
        self,
        sent: Tensor,
        langs: Optional[List[str]] = None,
        segment: Optional[Tensor] = None,
        model: Optional[transformers.PreTrainedModel] = None,
        return_raw_hidden_states: bool = False,
    ):
        if model is None:
            model = self.model
        mask = self.get_mask(sent)
        if isinstance(model, transformers.BertModel) or isinstance(
            self.model, transformers.RobertaModel
        ):
            output = model(input_ids=sent, attention_mask=mask, token_type_ids=segment)
        elif isinstance(model, transformers.XLMModel):
            lang_ids: Optional[torch.Tensor]
            if langs is not None:
                try:
                    batch_size, seq_len = sent.shape
                    lang_ids = torch.tensor(
                        [self.tokenizer.lang2id[lang] for lang in langs],
                        dtype=torch.long,
                        device=sent.device,
                    )
                    lang_ids = lang_ids.unsqueeze(1).expand(batch_size, seq_len)
                except KeyError as e:
                    print(f"KeyError with missing language {e}")
                    lang_ids = None
            output = model(
                input_ids=sent,
                attention_mask=mask,
                langs=lang_ids,
                token_type_ids=segment,
            )
        else:
            raise ValueError("Unsupported model")

        if return_raw_hidden_states:
            return output["hidden_states"]

        hs = self.map_feature(output["hidden_states"], langs)
        hs = self.process_feature(hs)
        hs = self.dropout(hs)
        hs = self.projector(hs, mask)
        return hs

    def map_feature(self, hidden_states: List[Tensor], langs):
        if self.mapping is None:
            return hidden_states

        assert len(set(langs)) == 1, "a batch should contain only one language"
        lang = langs[0]
        lang = constant.LANGUAGE_TO_ISO639.get(lang, lang)
        if lang not in self.mapping:
            return hidden_states

        hs = []
        for h, m in zip(hidden_states, self.mapping[lang]):
            hs.append(m(h))
        return hs

    def process_feature(self, hidden_states: List[Tensor]):
        if self.hparams.weighted_feature:
            hs: Tensor = torch.stack(hidden_states)
            weight = F.softmax(self.weight, dim=0).view(-1, 1, 1, 1)
            hs = hs * weight
            hs = hs.sum(dim=0)
        else:
            hs = hidden_states[self.hparams.feature_layer]
        return hs

    def evaluation_step_helper(self, batch, prefix) -> Dict[str, Tensor]:
        raise NotImplementedError

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.evaluation_step_helper(batch, "val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.evaluation_step_helper(batch, "tst")

    def training_epoch_end(self, outputs):
        return

    def aggregate_outputs(
        self, outputs: List[List[Dict[str, Tensor]]], langs: List[str], prefix: str
    ):
        assert prefix in ["val", "tst"]
        aver_result = defaultdict(list)
        for lang, output in zip(langs, outputs):
            for key in output[0]:
                mean_val = torch.stack([x[key] for x in output]).mean()
                self.log(key, mean_val)

                raw_key = key.replace(f"{lang}_", "")
                aver_result[raw_key].append(mean_val)

        for key, vals in aver_result.items():
            self.log(key, torch.stack(vals).mean())

    def aggregate_metrics(self, langs: List[str], prefix: str):
        aver_metric = defaultdict(list)
        for lang in langs:
            metric = self.metrics[lang]
            for key, val in metric.get_metric().items():
                self.log(f"{prefix}_{lang}_{key}", val)

                aver_metric[key].append(val)

        for key, vals in aver_metric.items():
            self.log(f"{prefix}_{key}", torch.stack(vals).mean())

    def validation_epoch_end(self, outputs):
        if len(self.hparams.val_langs) == 1:
            outputs = [outputs]
        self.aggregate_outputs(outputs, self.hparams.val_langs, "val")
        self.aggregate_metrics(self.hparams.val_langs, "val")
        return

    def test_epoch_end(self, outputs):
        if len(self.hparams.tst_langs) == 1:
            outputs = [outputs]
        self.aggregate_outputs(outputs, self.hparams.tst_langs, "tst")
        self.aggregate_metrics(self.hparams.tst_langs, "tst")
        return

    def get_warmup_and_total_steps(self):
        if self.hparams.max_steps is not None:
            max_steps = self.hparams.max_steps
        else:
            max_steps = self.hparams.max_epochs * self.batch_per_epoch
        assert not (
            self.hparams.warmup_steps != -1 and self.hparams.warmup_portion != -1
        )
        if self.hparams.warmup_steps != -1:
            assert self.hparams.warmup_steps > 0
            warmup_steps = self.hparams.warmup_steps
        elif self.hparams.warmup_portion != -1:
            assert 0 < self.hparams.warmup_portion < 1
            warmup_steps = int(max_steps * self.hparams.warmup_portion)
        else:
            warmup_steps = 1
        return warmup_steps, max_steps

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = []
        optimizer_grouped_parameters.append(
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            }
        )
        optimizer_grouped_parameters.append(
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            }
        )

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            betas=(0.9, self.hparams.adam_beta2),
            eps=self.hparams.adam_eps,
        )

        warmup_steps, max_steps = self.get_warmup_and_total_steps()
        if self.hparams.schedule == Schedule.invsqroot:
            scheduler = util.get_inverse_square_root_schedule_with_warmup(
                optimizer, warmup_steps
            )
            interval = "step"
        elif self.hparams.schedule == Schedule.linear:
            scheduler = util.get_linear_schedule_with_warmup(
                optimizer, warmup_steps, max_steps
            )
            interval = "step"
        elif self.hparams.schedule == Schedule.reduceOnPlateau:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.5, patience=0, min_lr=1e-6, mode="min"
            )
            interval = "epoch"
        else:
            raise ValueError(self.hparams.schedule)

        self.optimizer = optimizer
        self.scheduler = scheduler
        scheduler_dict = {"scheduler": scheduler, "interval": interval}
        if self.hparams.schedule == Schedule.reduceOnPlateau:
            scheduler_dict["monitor"] = "val_loss"
        return [optimizer], [scheduler_dict]

    def _get_signature(self, params: Dict):
        def md5_helper(obj):
            return hashlib.md5(str(obj).encode()).hexdigest()

        signature = dict()
        for key, val in params.items():
            if key == "tokenizer" and isinstance(val, transformers.PreTrainedTokenizer):
                signature[key] = md5_helper(list(val.get_vocab().items()))
            else:
                signature[key] = str(val)

        md5 = md5_helper(list(signature.items()))
        return md5, signature

    def prepare_datasets(self, split: str) -> List[Dataset]:
        raise NotImplementedError

    def prepare_datasets_helper(
        self,
        data_class: Type[Dataset],
        langs: List[str],
        split: str,
        max_len: int,
        **kwargs,
    ):
        datasets = []
        for lang in langs:
            filepath = data_class.get_file(self.hparams.data_dir, lang, split)
            if filepath is None:
                print(f"skipping {split} language: {lang}")
                continue
            params = {}
            params["task"] = self.hparams.task
            params["tokenizer"] = self.tokenizer
            params["filepath"] = filepath
            params["lang"] = lang
            params["split"] = split
            params["max_len"] = max_len
            if split == Split.train:
                params["subset_ratio"] = self.hparams.subset_ratio
                params["subset_count"] = self.hparams.subset_count
                params["subset_seed"] = self.hparams.subset_seed
            params.update(kwargs)
            md5, signature = self._get_signature(params)
            del params["task"]
            cache_file = f"{self.hparams.cache_path}/{md5}"
            if self.hparams.cache_dataset and os.path.isfile(cache_file):
                print(f"load from cache {filepath} with {self.hparams.pretrain}")
                dataset = torch.load(cache_file)
            else:
                dataset = data_class(**params)
                if self.hparams.cache_dataset:
                    print(f"save to cache {filepath} with {self.hparams.pretrain}")
                    torch.save(dataset, cache_file)
                    with open(f"{cache_file}.json", "w") as fp:
                        json.dump(signature, fp)
            datasets.append(dataset)
        return datasets

    def train_dataloader(self):
        if self.trn_datasets is None:
            self.trn_datasets = self.prepare_datasets(Split.train)

        collate = partial(util.default_collate, padding=self.padding)
        if len(self.trn_datasets) == 1:
            dataset = self.trn_datasets[0]
            sampler = RandomSampler(dataset)
        else:
            dataset = ConcatDataset(self.trn_datasets)
            if self.hparams.mix_sampling:
                sampler = RandomSampler(dataset)
            else:
                sampler = util.ConcatSampler(dataset, self.hparams.batch_size)

        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            sampler=sampler,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate,
            num_workers=1,
        )

    def val_dataloader(self):
        if self.val_datasets is None:
            self.val_datasets = self.prepare_datasets(Split.dev)

        collate = partial(util.default_collate, padding=self.padding)
        return [
            DataLoader(
                val_dataset,
                batch_size=self.hparams.eval_batch_size,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate,
                num_workers=1,
            )
            for val_dataset in self.val_datasets
        ]

    def test_dataloader(self):
        if self.tst_datasets is None:
            self.tst_datasets = self.prepare_datasets(Split.test)

        collate = partial(util.default_collate, padding=self.padding)
        return [
            DataLoader(
                tst_dataset,
                batch_size=self.hparams.eval_batch_size,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate,
                num_workers=1,
            )
            for tst_dataset in self.tst_datasets
        ]

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # fmt: off
        # shared
        parser.add_argument("--task", required=True, choices=Task().choices(), type=str)
        parser.add_argument("--data_dir", required=True, type=str)
        parser.add_argument("--trn_langs", required=True, nargs="+", type=str)
        parser.add_argument("--val_langs", required=True, nargs="+", type=str)
        parser.add_argument("--tst_langs", default=[], nargs="*", type=str)
        parser.add_argument("--max_trn_len", default=128, type=int)
        parser.add_argument("--max_tst_len", default=128, type=int)
        parser.add_argument("--subset_ratio", default=1.0, type=float)
        parser.add_argument("--subset_count", default=-1, type=int)
        parser.add_argument("--subset_seed", default=42, type=int)
        parser.add_argument("--mix_sampling", default=False, type=util.str2bool)
        # encoder
        parser.add_argument("--pretrain", required=True, type=str)
        parser.add_argument("--freeze_layer", default=-1, type=int)
        parser.add_argument("--feature_layer", default=-1, type=int)
        parser.add_argument("--weighted_feature", default=False, type=util.str2bool)
        parser.add_argument("--projector", default="id", choices=["id", "meanpool", "transformer"], type=str)
        parser.add_argument("--projector_trm_hidden_size", default=3072, type=int)
        parser.add_argument("--projector_trm_num_heads", default=12, type=int)
        parser.add_argument("--projector_trm_num_layers", default=4, type=int)
        parser.add_argument("--projector_dropout", default=0.2, type=float)
        parser.add_argument("--input_dropout", default=0.2, type=float)
        parser.add_argument("--mapping", default="", type=str)
        # misc
        parser.add_argument("--seed", default=42, type=int)
        parser.add_argument("--learning_rate", default=5e-5, type=float)
        parser.add_argument("--adam_beta2", default=0.99, type=float)
        parser.add_argument("--adam_eps", default=1e-8, type=float)
        parser.add_argument("--weight_decay", default=0.0, type=float)
        parser.add_argument("--batch_size", default=32, type=int)
        parser.add_argument("--eval_batch_size", default=32, type=int)
        parser.add_argument("--schedule", default=Schedule.linear, choices=Schedule().choices(), type=str)
        parser.add_argument("--warmup_steps", default=-1, type=int)
        parser.add_argument("--warmup_portion", default=-1, type=float)
        # fmt: on
        return parser
