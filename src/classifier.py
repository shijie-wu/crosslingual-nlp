from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from base_model import Model
from dataset import LanguageID, MLDoc, Xnli
from enumeration import Split, Task
from metric import AccuracyMetric


class Classifier(Model):
    def __init__(self, hparams):
        super(Classifier, self).__init__(hparams)

        self._comparsion = {Task.xnli: "max", Task.mldoc: "max", Task.langid: "max"}[
            self.hparams.task
        ]
        self._selection_criterion = {
            Task.xnli: "val_acc",
            Task.mldoc: "val_acc",
            Task.langid: "val_acc",
        }[self.hparams.task]
        self._nb_labels: Optional[int] = None
        self._nb_labels = {
            Task.xnli: Xnli.nb_labels(),
            Task.mldoc: MLDoc.nb_labels(),
            Task.langid: LanguageID.nb_labels(),
        }[self.hparams.task]
        self._metric = {
            Task.xnli: AccuracyMetric(),
            Task.mldoc: AccuracyMetric(),
            Task.langid: AccuracyMetric(),
        }[self.hparams.task]

        self.classifier = nn.Linear(self.hidden_size, self.nb_labels)
        self.padding = {
            "sent": self.tokenizer.pad_token_id,
            "segment": 0,
            "label": 0,
            "lang": 0,
        }

        self.setup_metrics()

    @property
    def nb_labels(self):
        assert self._nb_labels is not None
        return self._nb_labels

    def preprocess_batch(self, batch):
        batch["label"] = batch["label"].view(-1)
        return batch

    def forward(self, batch):
        batch = self.preprocess_batch(batch)
        hs = self.encode_sent(batch["sent"], batch["lang"], segment=batch["segment"])
        if hs.dim() == 3:
            hs = hs[:, 0]
        logits = self.classifier(hs)
        log_probs = F.log_softmax(logits, dim=-1)

        loss = F.nll_loss(log_probs.view(-1, self.nb_labels), batch["label"])
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

    def eval_helper(self, batch, prefix):
        loss, log_probs = self.forward(batch)

        assert (
            len(set(batch["lang"])) == 1
        ), "eval batch should contain only one language"
        lang = batch["lang"][0]
        self.metrics[lang].add(batch["label"], log_probs)

        result = dict()
        result[f"{prefix}_{lang}_loss"] = loss.view(1)
        return result

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.eval_helper(batch, "val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.eval_helper(batch, "tst")

    def prepare_data(self):
        hparams = self.hparams
        if hparams.task == Task.xnli:
            data_class = Xnli
        elif hparams.task == Task.mldoc:
            data_class = MLDoc
        elif hparams.task == Task.langid:
            data_class = LanguageID
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
