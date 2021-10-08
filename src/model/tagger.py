from copy import deepcopy
from typing import List, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

import util
from dataset import LABEL_PAD_ID, ConllNER, Dataset, UdPOS, WikiAnnNER
from enumeration import Split, Task
from metric import NERMetric, POSMetric, convert_bio_to_spans
from model.base import Model
from model.crf import ChainCRF


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

        self.id2label = {
            Task.conllner: ConllNER.get_labels(),
            Task.wikiner: WikiAnnNER.get_labels(),
            Task.udpos: UdPOS.get_labels(),
        }[self.hparams.task]

        if self.hparams.tagger_use_crf:
            self.crf = ChainCRF(self.hidden_size, self.nb_labels, bigram=True)
        else:
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
        return batch

    def predict(self, batch):
        sent = batch["sent"]
        lang = batch["lang"]
        first_subword_mask = batch["first_subword_mask"]

        hs = self.encode_sent(sent, lang)
        if self.hparams.tagger_use_crf:
            mask = first_subword_mask.float()
            energy = self.crf(hs, mask=mask)
            predicted_labels = self.crf.decode(energy, mask=mask)
        else:
            logits = self.classifier(hs)
            log_probs = F.log_softmax(logits, dim=-1)
            _, predicted_labels = torch.max(log_probs, dim=-1)

        predicted_labels = predicted_labels * first_subword_mask

        predicted_labels = predicted_labels.cpu().numpy()
        first_subword_mask = first_subword_mask.cpu().numpy()

        prediction = []
        for i in range(len(sent)):
            labels = []
            for j in range(len(sent[i])):
                if first_subword_mask[i, j] != 1:
                    continue
                labels.append(self.id2label[predicted_labels[i, j]])
            if self.hparams.task in [Task.conllner, Task.wikiner]:
                clean_labels: List[str] = deepcopy(labels)
                clean_spans = convert_bio_to_spans(labels)
                for entity, start, end in clean_spans:
                    for k, pos in enumerate(range(start, end)):
                        if k == 0:
                            clean_labels[pos] = f"B-{entity}"
                        else:
                            clean_labels[pos] = f"I-{entity}"
                labels = clean_labels
            assert len(batch["orig_sent"][i]) == len(labels)
            prediction.append(
                {
                    "sent": batch["orig_sent"][i],
                    "labels": labels,
                }
            )
        return prediction

    def forward(self, batch):
        batch = self.preprocess_batch(batch)
        hs = self.encode_sent(batch["sent"], batch["lang"])
        if self.hparams.tagger_use_crf:
            mask = (batch["labels"] != LABEL_PAD_ID).float()
            energy = self.crf(hs, mask=mask)
            target = batch["labels"].masked_fill(
                batch["labels"] == LABEL_PAD_ID, self.nb_labels
            )
            loss = self.crf.loss(energy, target, mask=mask)
            log_probs = energy
        else:
            logits = self.classifier(hs)
            log_probs = F.log_softmax(logits, dim=-1)

            loss = F.nll_loss(
                log_probs.view(-1, self.nb_labels),
                batch["labels"].view(-1),
                ignore_index=LABEL_PAD_ID,
            )
        return loss, log_probs

    def training_step(self, batch, batch_idx):
        loss, _ = self.forward(batch)
        self.log("loss", loss)
        return loss

    def evaluation_step_helper(self, batch, prefix):
        loss, log_probs = self.forward(batch)

        assert (
            len(set(batch["lang"])) == 1
        ), "eval batch should contain only one language"
        lang = batch["lang"][0]
        if self.hparams.tagger_use_crf:
            energy = log_probs
            prediction = self.crf.decode(
                energy, mask=(batch["labels"] != LABEL_PAD_ID).float()
            )
            self.metrics[lang].add(batch["labels"], prediction)
        else:
            self.metrics[lang].add(batch["labels"], log_probs)

        result = dict()
        result[f"{prefix}_{lang}_loss"] = loss
        return result

    def prepare_datasets(self, split: str) -> List[Dataset]:
        hparams = self.hparams
        data_class: Type[Dataset]
        if hparams.task == Task.conllner:
            data_class = ConllNER
        elif hparams.task == Task.wikiner:
            data_class = WikiAnnNER
        elif hparams.task == Task.udpos:
            data_class = UdPOS
        else:
            raise ValueError(f"Unsupported task: {hparams.task}")

        if split == Split.train:
            return self.prepare_datasets_helper(
                data_class, hparams.trn_langs, Split.train, hparams.max_trn_len
            )
        elif split == Split.dev:
            return self.prepare_datasets_helper(
                data_class, hparams.val_langs, Split.dev, hparams.max_tst_len
            )
        elif split == Split.test:
            return self.prepare_datasets_helper(
                data_class, hparams.tst_langs, Split.test, hparams.max_tst_len
            )
        else:
            raise ValueError(f"Unsupported split: {hparams.split}")

    @classmethod
    def add_model_specific_args(cls, parser):
        parser.add_argument("--tagger_use_crf", default=False, type=util.str2bool)
        return parser
