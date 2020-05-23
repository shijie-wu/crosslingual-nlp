from copy import deepcopy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metric import NERMetric, POSMetric
from base_model import Model
from dataset import LABEL_PAD_ID, ConllNER, UdPOS, WikiAnnNER
from enumeration import Split, Task


class Tagger(Model):
    def __init__(self, hparams):
        super(Tagger, self).__init__(hparams)

        self._comparsion = {
            Task.conllner: "max",
            Task.wikiner: "max",
            Task.udpos: "max",
        }[self.hparams.task]
        self._selection_criterion = {
            Task.conllner: "val_f1",
            Task.wikiner: "val_f1",
            Task.udpos: "val_acc",
        }[self.hparams.task]
        self._nb_labels: Optional[int] = None
        self._nb_labels = {
            Task.conllner: ConllNER.nb_labels(),
            Task.wikiner: WikiAnnNER.nb_labels(),
            Task.udpos: UdPOS.nb_labels(),
        }[self.hparams.task]
        self._metric = {
            Task.conllner: NERMetric(ConllNER.get_labels()),
            Task.wikiner: NERMetric(WikiAnnNER.get_labels()),
            Task.udpos: POSMetric(),
        }[self.hparams.task]

        self.classifier = nn.Linear(self.hidden_size, self.nb_labels)
        self.padding = {
            "sent": self.tokenizer.pad_token_id,
            "lang": 0,
            "labels": LABEL_PAD_ID,
        }

        self.setup_metrics()

    @property
    def nb_labels(self):
        assert self._nb_labels is not None
        return self._nb_labels

    def preprocess_batch(self, batch):
        assert len(set(batch["lang"])) == 1
        batch["lang"] = batch["lang"][0]
        return batch

    def forward(self, batch):
        batch = self.preprocess_batch(batch)
        hs = self.encode_sent(batch["sent"], batch["lang"])
        logits = self.classifier(hs)
        log_probs = F.log_softmax(logits, dim=-1)

        loss = F.nll_loss(
            log_probs.view(-1, self.nb_labels),
            batch["labels"].view(-1),
            ignore_index=LABEL_PAD_ID,
        )
        return loss, log_probs

    def training_step(self, batch, batch_idx):
        result = {"lr": self.get_lr()}

        loss, log_probs = self.forward(batch)
        result["loss"] = loss
        return {
            "loss": result["loss"],
            "progress_bar": {"lr": result["lr"]},
            "log": result,
        }

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, log_probs = self.forward(batch)

        self.metrics[batch["lang"]].add(batch["labels"], log_probs)

        result = dict()
        lang = batch["lang"]
        result[f"val_{lang}_loss"] = loss.view(1)
        return result

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, log_probs = self.forward(batch)

        self.metrics[batch["lang"]].add(batch["labels"], log_probs)

        result = dict()
        lang = batch["lang"]
        result[f"tst_{lang}_loss"] = loss.view(1)
        return result

    def prepare_data(self):
        hparams = self.hparams
        if hparams.task == Task.conllner:
            data_class = ConllNER
        elif hparams.task == Task.wikiner:
            data_class = WikiAnnNER
        elif hparams.task == Task.udpos:
            data_class = UdPOS
        else:
            raise ValueError(f"Unsupported task: {hparams.task}")

        self.trn_datasets = self.prepare_datasets(
            data_class, hparams.trn_langs, Split.train, hparams.max_trn_len
        )
        self.val_datasets = self.prepare_datasets(
            data_class, hparams.val_langs, Split.dev, hparams.max_tst_len
        )
        self.tst_datasets = self.prepare_datasets(
            data_class, hparams.tst_langs, Split.test, hparams.max_tst_len
        )

    @classmethod
    def add_model_specific_args(cls, parser):
        return parser
