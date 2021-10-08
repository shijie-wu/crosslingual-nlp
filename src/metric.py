import re
from typing import Dict, List

import torch
from sklearn import metrics

LABEL_PAD_ID = -1


def convert_bio_to_spans(bio_sequence):
    spans = []  # (label, startindex, endindex)
    cur_start = None
    cur_label = None
    N = len(bio_sequence)
    for t in range(N + 1):
        if (cur_start is not None) and (t == N or re.search("^[BO]", bio_sequence[t])):
            assert cur_label is not None
            spans.append((cur_label, cur_start, t))
            cur_start = None
            cur_label = None
        if t == N:
            continue
        assert bio_sequence[t] and bio_sequence[t][0] in ("B", "I", "O")
        if bio_sequence[t].startswith("B"):
            cur_start = t
            cur_label = re.sub("^B-?", "", bio_sequence[t]).strip()
        if bio_sequence[t].startswith("I"):
            if cur_start is None:
                # warning(
                #     "BIO inconsistency: I without starting B. Rewriting to B.")
                newseq = bio_sequence[:]
                newseq[t] = "B" + newseq[t][1:]
                return convert_bio_to_spans(newseq)
            continuation_label = re.sub("^I-?", "", bio_sequence[t])
            if continuation_label != cur_label:
                newseq = bio_sequence[:]
                newseq[t] = "B" + newseq[t][1:]
                # warning(
                #     "BIO inconsistency: %s but current label is '%s'. Rewriting to %s"
                #     % (bio_sequence[t], cur_label, newseq[t]))
                return convert_bio_to_spans(newseq)

    # should have exited for last span ending at end by now
    assert cur_start is None
    return spans


def to_tensor(wrapped_func):
    def func(*args, **kwargs):
        result = wrapped_func(*args, **kwargs)
        return {k: torch.tensor(v, dtype=torch.float) for k, v in result.items()}

    return func


class Metric(object):
    def add(self, gold, prediction):
        raise NotImplementedError

    def get_metric(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @staticmethod
    def unpack(*tensors: torch.Tensor):
        return (x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in tensors)


class BinaryClassificationMetric(Metric):
    def __init__(self, average="binary", pos_label=1):
        self.average = average
        self.pos_label = pos_label
        self.gold = []
        self.prediction = []

    def add(self, gold, prediction):
        gold, prediction = self.unpack(gold, prediction)
        _, prediction = torch.max(prediction, dim=1)
        self.gold.extend(gold.tolist())
        self.prediction.extend(prediction.tolist())

    @to_tensor
    def get_metric(self):
        acc = metrics.accuracy_score(self.gold, self.prediction)
        recall = metrics.recall_score(
            self.gold, self.prediction, average=self.average, pos_label=self.pos_label
        )
        precision = metrics.precision_score(
            self.gold, self.prediction, average=self.average, pos_label=self.pos_label
        )
        f1 = metrics.f1_score(
            self.gold, self.prediction, average=self.average, pos_label=self.pos_label
        )
        return {
            "acc": acc * 100,
            "recall": recall * 100,
            "precision": precision * 100,
            "f1": f1 * 100,
        }

    def reset(self):
        self.gold = []
        self.prediction = []


class AccuracyMetric(Metric):
    def __init__(self):
        self.gold = []
        self.prediction = []

    def add(self, gold, prediction):
        gold, prediction = self.unpack(gold, prediction)
        _, prediction = torch.max(prediction, dim=1)
        self.gold.extend(gold.tolist())
        self.prediction.extend(prediction.tolist())

    @to_tensor
    def get_metric(self):
        acc = metrics.accuracy_score(self.gold, self.prediction)
        # check nan
        if float(acc) != acc:
            acc = 0.0
        return {"acc": acc * 100}

    def reset(self):
        self.gold = []
        self.prediction = []


class POSMetric(Metric):
    def __init__(self):
        self.num_correct = 0
        self.num_tokens = 0

    def add(self, gold, prediction):
        """
        gold is label
        prediction is logits
        """
        gold, prediction = self.unpack(gold, prediction)
        _, prediction = torch.max(prediction, dim=-1)
        bs, seq_len = prediction.shape
        for ii in range(bs):
            for jj in range(seq_len):
                gold_label, pred_label = gold[ii, jj], prediction[ii, jj]
                if gold_label == LABEL_PAD_ID:
                    continue
                if gold_label == pred_label:
                    self.num_correct += 1
                self.num_tokens += 1

    @to_tensor
    def get_metric(self):
        try:
            acc = self.num_correct / self.num_tokens
        except ZeroDivisionError:
            acc = 0
        return {"acc": acc * 100}

    def reset(self):
        self.num_correct = 0
        self.num_tokens = 0


class NERMetric(Metric):
    def __init__(self, label: List[str]):
        self.label = label

        self.tp, self.fp, self.fn = 0, 0, 0

    def add(self, gold, prediction):
        """
        gold is label
        prediction is logits (batch_size, seq_len, num_labels) or prediction (batch_size,
        seq_len)
        """
        gold, prediction = self.unpack(gold, prediction)
        if prediction.dim() > 2:
            _, prediction = torch.max(prediction, dim=-1)
        bs, seq_len = prediction.shape
        for ii in range(bs):
            goldseq, predseq = [], []
            for jj in range(seq_len):
                gold_label, pred_label = gold[ii, jj], prediction[ii, jj]
                if gold_label == LABEL_PAD_ID:
                    continue
                goldseq.append(self.label[gold_label])
                predseq.append(self.label[pred_label])

            goldspans = convert_bio_to_spans(goldseq)
            predspans = convert_bio_to_spans(predseq)

            goldspans_set = set(goldspans)
            predspans_set = set(predspans)

            # tp: number of spans that gold and pred have
            # fp: number of spans that pred had that gold didn't (incorrect predictions)
            # fn: number of spans that gold had that pred didn't (didn't recall)
            self.tp += len(goldspans_set & predspans_set)
            self.fp += len(predspans_set - goldspans_set)
            self.fn += len(goldspans_set - predspans_set)

    @to_tensor
    def get_metric(self):
        try:
            prec = self.tp / (self.tp + self.fp)
            rec = self.tp / (self.tp + self.fn)
            f1 = 2 * prec * rec / (prec + rec)
        except ZeroDivisionError:
            f1 = 0
        return {"f1": f1 * 100}

    def reset(self):
        self.tp, self.fp, self.fn = 0, 0, 0


class ParsingMetric(Metric):
    """
    from allennlp.training.metrics.AttachmentScores

    Computes labeled and unlabeled attachment scores for a dependency parse, as well as
    sentence level exact match for both labeled and unlabeled trees. Note that the input
    to this metric is the sampled predictions, not the distribution itself.

    Parameters
    ----------
    ignore_classes : ``List[int]``, optional (default = None)
        A list of label ids to ignore when computing metrics.
    """

    def __init__(self, ignore_classes: List[int] = None):
        self._labeled_correct = 0.0
        self._unlabeled_correct = 0.0
        self._exact_labeled_correct = 0.0
        self._exact_unlabeled_correct = 0.0
        self._total_words = 0.0
        self._total_sentences = 0.0

        self._ignore_classes: List[int] = ignore_classes or []

    def add(  # type: ignore
        self,
        gold_indices: torch.Tensor,
        gold_labels: torch.Tensor,
        predicted_indices: torch.Tensor,
        predicted_labels: torch.Tensor,
        mask: torch.Tensor,
    ):
        """
        Parameters
        ----------
        predicted_indices : ``torch.Tensor``, required.
            A tensor of head index predictions of shape (batch_size, timesteps).
        predicted_labels : ``torch.Tensor``, required.
            A tensor of arc label predictions of shape (batch_size, timesteps).
        gold_indices : ``torch.Tensor``, required.
            A tensor of the same shape as ``predicted_indices``.
        gold_labels : ``torch.Tensor``, required.
            A tensor of the same shape as ``predicted_labels``.
        mask: ``torch.Tensor``, required.
            A tensor of the same shape as ``predicted_indices``.
        """
        unwrapped = self.unpack(
            predicted_indices, predicted_labels, gold_indices, gold_labels, mask
        )
        predicted_indices, predicted_labels, gold_indices, gold_labels, mask = unwrapped

        mask = mask.long()
        predicted_indices = predicted_indices.long()
        predicted_labels = predicted_labels.long()
        gold_indices = gold_indices.long()
        gold_labels = gold_labels.long()

        # Multiply by a mask denoting locations of
        # gold labels which we should ignore.
        for label in self._ignore_classes:
            label_mask = gold_labels.eq(label)
            mask = mask * (~label_mask).long()

        correct_indices = predicted_indices.eq(gold_indices).long() * mask
        unlabeled_exact_match = (correct_indices + (1 - mask)).prod(dim=-1)
        correct_labels = predicted_labels.eq(gold_labels).long() * mask
        correct_labels_and_indices = correct_indices * correct_labels
        labeled_exact_match = (correct_labels_and_indices + (1 - mask)).prod(dim=-1)

        self._unlabeled_correct += correct_indices.sum().item()
        self._exact_unlabeled_correct += unlabeled_exact_match.sum().item()
        self._labeled_correct += correct_labels_and_indices.sum().item()
        self._exact_labeled_correct += labeled_exact_match.sum().item()
        self._total_sentences += correct_indices.size(0)
        self._total_words += correct_indices.numel() - (1 - mask).sum().item()

    @to_tensor
    def get_metric(self):
        unlabeled_attachment_score = 0.0
        labeled_attachment_score = 0.0
        unlabeled_exact_match = 0.0
        labeled_exact_match = 0.0
        if self._total_words > 0.0:
            unlabeled_attachment_score = self._unlabeled_correct / self._total_words
            labeled_attachment_score = self._labeled_correct / self._total_words
        if self._total_sentences > 0:
            unlabeled_exact_match = (
                self._exact_unlabeled_correct / self._total_sentences
            )
            labeled_exact_match = self._exact_labeled_correct / self._total_sentences
        return {
            "uas": unlabeled_attachment_score * 100,
            "las": labeled_attachment_score * 100,
            "uem": unlabeled_exact_match * 100,
            "lem": labeled_exact_match * 100,
        }

    def reset(self):
        self._labeled_correct = 0.0
        self._unlabeled_correct = 0.0
        self._exact_labeled_correct = 0.0
        self._exact_unlabeled_correct = 0.0
        self._total_words = 0.0
        self._total_sentences = 0.0
