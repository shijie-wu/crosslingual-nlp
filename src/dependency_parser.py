from typing import List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from base_model import Model
from chu_liu_edmonds import decode_mst
from dataset import LABEL_PAD_ID, Dataset, ParsingDataset
from enumeration import Split, Task
from metric import ParsingMetric
from module import BilinearAttention, InputVariationalDropout
from util import masked_log_softmax, str2bool

POS_TO_IGNORE = {"``", "''", ":", ",", ".", "PU", "PUNCT", "SYM"}


class DependencyParser(Model):
    def __init__(self, hparams):
        super(DependencyParser, self).__init__(hparams)

        self._comparsion = {Task.parsing: "max"}[self.hparams.task]
        self._selection_criterion = {Task.parsing: "val_las"}[self.hparams.task]
        self._nb_labels: Optional[int] = None
        self._nb_labels = {Task.parsing: ParsingDataset.nb_labels()}[self.hparams.task]
        self._nb_pos_tags: Optional[int] = None
        self._nb_pos_tags = {Task.parsing: ParsingDataset.nb_pos_tags()}[
            self.hparams.task
        ]
        self._metric = {Task.parsing: ParsingMetric()}[self.hparams.task]

        encode_dim = self.hidden_size
        hparams = self.hparams
        assert not (hparams.parser_use_pos and hparams.parser_use_predict_pos)
        if hparams.parser_use_pos or hparams.parser_use_predict_pos:
            if hparams.parser_use_pos:
                nb_pos_tags = self.nb_pos_tags + 1
                padding_idx = -1
            elif hparams.parser_use_predict_pos:
                assert (
                    hparams.parser_predict_pos
                ), "when parser_use_predict_pos is True, parser_predict_pos should also be True"
                nb_pos_tags = self.nb_pos_tags
                padding_idx = None
            else:
                raise ValueError(
                    "parser_use_pos and parser_use_predict_pos are mutually exclusive"
                )
            self.pos_embed = nn.Embedding(
                nb_pos_tags, hparams.parser_pos_dim, padding_idx=padding_idx
            )
            encode_dim += hparams.parser_pos_dim

        if hparams.parser_predict_pos:
            self.pos_tagger = nn.Linear(self.hidden_size, self.nb_pos_tags)

        self.head_arc_ff = self._ff(
            encode_dim, hparams.parser_arc_dim, hparams.parser_dropout
        )
        self.child_arc_ff = self._ff(
            encode_dim, hparams.parser_arc_dim, hparams.parser_dropout
        )
        self.arc_attention = BilinearAttention(
            hparams.parser_arc_dim, hparams.parser_arc_dim, use_input_biases=True
        )

        self.head_tag_ff = self._ff(
            encode_dim, hparams.parser_tag_dim, hparams.parser_dropout
        )
        self.child_tag_ff = self._ff(
            encode_dim, hparams.parser_tag_dim, hparams.parser_dropout
        )
        self.tag_bilinear = nn.Bilinear(
            hparams.parser_tag_dim, hparams.parser_tag_dim, self.nb_labels
        )

        punctuation_tag_indices = {
            pos_tag: index
            for index, pos_tag in enumerate(ParsingDataset.get_pos_tags())
            if pos_tag in POS_TO_IGNORE
        }
        self._pos_to_ignore = set(punctuation_tag_indices.values())
        print(
            f"Found POS tags corresponding to the following punctuation :"
            f"{punctuation_tag_indices}, ignoring words with these POS tags for"
            f"evaluation."
        )
        self.padding = {
            "sent": self.tokenizer.pad_token_id,
            "lang": 0,
            "pos_tags": LABEL_PAD_ID,
            "heads": -1,
            "labels": LABEL_PAD_ID,
        }

        self.setup_metrics()

    def _ff(self, input_dim, output_dim, dropout):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ELU(),
            InputVariationalDropout(dropout),
        )

    @property
    def nb_labels(self):
        assert self._nb_labels is not None
        return self._nb_labels

    @property
    def nb_pos_tags(self):
        assert self._nb_pos_tags is not None
        return self._nb_pos_tags

    def preprocess_batch(self, batch):
        _, seq_len = batch["sent"].shape
        batch["first_subword_mask"] = batch["heads"] != -1
        batch["heads"] = batch["heads"].masked_fill(batch["heads"] >= seq_len, -1)
        return batch

    def forward(self, batch):
        batch = self.preprocess_batch(batch)
        sent = batch["sent"]
        lang = batch["lang"]
        pos_tags = batch["pos_tags"]
        heads = batch["heads"]
        labels = batch["labels"]
        first_subword_mask = batch["first_subword_mask"]

        hs = self.encode_sent(sent, lang)
        if self.hparams.parser_predict_pos:
            logits = self.pos_tagger(hs)
            log_probs = F.log_softmax(logits, dim=-1)
            pos_nll = F.nll_loss(
                log_probs.view(-1, self.nb_pos_tags),
                batch["pos_tags"].view(-1),
                ignore_index=LABEL_PAD_ID,
            )
        else:
            log_probs = None
            pos_nll = 0

        if self.hparams.parser_use_pos:
            hs_pos = self.pos_embed(
                pos_tags.masked_fill(pos_tags < 0, self.nb_pos_tags)
            )
            hs = torch.cat((hs, hs_pos), dim=-1)
        elif self.hparams.parser_use_predict_pos:
            assert log_probs is not None
            hs_pos = F.linear(log_probs.exp().detach(), self.pos_embed.weight.t())
            hs = torch.cat((hs, hs_pos), dim=-1)

        head_arc = self.head_arc_ff(hs)
        child_arc = self.child_arc_ff(hs)
        score_arc = self.arc_attention(head_arc, child_arc)

        head_tag = self.head_tag_ff(hs)
        child_tag = self.child_tag_ff(hs)

        minus_inf = -1e8
        minus_mask = ~first_subword_mask * minus_inf
        score_arc = score_arc + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        arc_nll, tag_nll = self._construct_loss(
            head_tag=head_tag,
            child_tag=child_tag,
            score_arc=score_arc,
            head_indices=heads,
            head_tags=labels,
            mask=first_subword_mask,
        )
        loss = arc_nll + tag_nll + pos_nll

        return loss, head_tag, child_tag, score_arc

    def _construct_loss(
        self,
        head_tag: torch.Tensor,
        child_tag: torch.Tensor,
        score_arc: torch.Tensor,
        head_indices: torch.Tensor,
        head_tags: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the arc and tag loss for a sequence given gold head indices and tags.

        # Parameters

        head_tag : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, tag_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag : `torch.Tensor`, required
            A tensor of shape (batch_size, sequence_length, tag_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        score_arc : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to
            generate a distribution over attachments of a given word to all other words.
        head_indices : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length).
            The indices of the heads for every word.
        head_tags : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length).
            The dependency labels of the heads for every word.
        mask : `torch.BoolTensor`, required.
            A mask of shape (batch_size, sequence_length), denoting unpadded
            elements in the sequence.

        # Returns

        arc_nll : `torch.Tensor`, required.
            The negative log likelihood from the arc loss.
        tag_nll : `torch.Tensor`, required.
            The negative log likelihood from the arc tag loss.
        """
        batch_size, sequence_length, _ = score_arc.size()
        # shape (batch_size, 1)
        range_vector = torch.arange(batch_size, device=score_arc.device).unsqueeze(1)
        # shape (batch_size, sequence_length, sequence_length)
        normalised_arc_logits = (
            masked_log_softmax(score_arc, mask) * mask.unsqueeze(2) * mask.unsqueeze(1)
        )

        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self._get_head_tags(head_tag, child_tag, head_indices)
        normalised_head_tag_logits = masked_log_softmax(
            head_tag_logits, mask.unsqueeze(-1)
        ) * mask.unsqueeze(-1)
        # index matrix with shape (batch, sequence_length)
        timestep_index = torch.arange(sequence_length, device=score_arc.device)
        child_index = (
            timestep_index.view(1, sequence_length)
            .expand(batch_size, sequence_length)
            .long()
        )
        # shape (batch_size, sequence_length)
        arc_loss = normalised_arc_logits[range_vector, child_index, head_indices]
        tag_loss = normalised_head_tag_logits[range_vector, child_index, head_tags]
        # We don't care about predictions for the symbolic ROOT token's head,
        # so we remove it from the loss.
        arc_loss = arc_loss[:, 1:]
        tag_loss = tag_loss[:, 1:]

        # The number of valid positions is equal to the number of unmasked elements minus
        # 1 per sequence in the batch, to account for the symbolic HEAD token.
        valid_positions = mask.sum() - batch_size

        arc_nll = -arc_loss.sum() / valid_positions.float()
        tag_nll = -tag_loss.sum() / valid_positions.float()
        return arc_nll, tag_nll

    def _greedy_decode(
        self,
        head_tag: torch.Tensor,
        child_tag: torch.Tensor,
        score_arc: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions by decoding the unlabeled arcs
        independently for each word and then again, predicting the head tags of
        these greedily chosen arcs independently. Note that this method of decoding
        is not guaranteed to produce trees (i.e. there maybe be multiple roots,
        or cycles when children are attached to their parents).

        # Parameters

        head_tag : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, tag_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag : `torch.Tensor`, required
            A tensor of shape (batch_size, sequence_length, tag_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        score_arc : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to
            generate a distribution over attachments of a given word to all other words.

        # Returns

        heads : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length) representing the
            greedily decoded heads of each word.
        head_tags : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length) representing the
            dependency tags of the greedily decoded heads of each word.
        """
        # Mask the diagonal, because the head of a word can't be itself.
        score_arc = score_arc + torch.diag(
            torch.zeros(mask.size(1), device=score_arc.device).fill_(-np.inf)
        )
        # Mask padded tokens, because we only want to consider actual words as heads.
        if mask is not None:
            minus_mask = ~mask.unsqueeze(2)
            score_arc.masked_fill_(minus_mask, -np.inf)

        # Compute the heads greedily.
        # shape (batch_size, sequence_length)
        _, heads = score_arc.max(dim=2)

        # Given the greedily predicted heads, decode their dependency tags.
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self._get_head_tags(head_tag, child_tag, heads)
        _, head_tags = head_tag_logits.max(dim=2)
        return heads, head_tags

    def _mst_decode(
        self,
        head_tag: torch.Tensor,
        child_tag: torch.Tensor,
        score_arc: torch.Tensor,
        mask: torch.BoolTensor,
        lengths: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions using the Edmonds' Algorithm
        for finding minimum spanning trees on directed graphs. Nodes in the
        graph are the words in the sentence, and between each pair of nodes,
        there is an edge in each direction, where the weight of the edge corresponds
        to the most likely dependency label probability for that arc. The MST is
        then generated from this directed graph.

        # Parameters

        head_tag : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, tag_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag : `torch.Tensor`, required
            A tensor of shape (batch_size, sequence_length, tag_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        score_arc : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to
            generate a distribution over attachments of a given word to all other words.

        # Returns

        heads : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length) representing the
            greedily decoded heads of each word.
        head_tags : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length) representing the
            dependency tags of the optimally decoded heads of each word.
        """
        batch_size, sequence_length, tag_dim = head_tag.size()

        # lengths = mask.data.sum(dim=1).long().cpu().numpy()

        expanded_shape = [batch_size, sequence_length, sequence_length, tag_dim]
        head_tag = head_tag.unsqueeze(2)
        head_tag = head_tag.expand(*expanded_shape).contiguous()
        child_tag = child_tag.unsqueeze(1)
        child_tag = child_tag.expand(*expanded_shape).contiguous()
        # Shape (batch_size, sequence_length, sequence_length, num_head_tags)
        pairwise_head_logits = self.tag_bilinear(head_tag, child_tag)

        # Note that this log_softmax is over the tag dimension, and we don't consider
        # pairs of tags which are invalid (e.g are a pair which includes a padded
        # element) anyway below. Shape (batch, num_labels,sequence_length,
        # sequence_length)
        normalized_pairwise_head_logits = F.log_softmax(
            pairwise_head_logits, dim=3
        ).permute(0, 3, 1, 2)

        # Mask padded tokens, because we only want to consider actual words as heads.
        minus_inf = -1e8
        minus_mask = ~mask * minus_inf
        score_arc = score_arc + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        # Shape (batch_size, sequence_length, sequence_length)
        normalized_arc_logits = F.log_softmax(score_arc, dim=2).transpose(1, 2)

        # Shape (batch_size, num_head_tags, sequence_length, sequence_length)
        # This energy tensor expresses the following relation:
        # energy[i,j] = "Score that i is the head of j". In this
        # case, we have heads pointing to their children.
        batch_energy = torch.exp(
            normalized_arc_logits.unsqueeze(1) + normalized_pairwise_head_logits
        )
        return self._run_mst_decoding(batch_energy, lengths)

    @staticmethod
    def _run_mst_decoding(
        batch_energy: torch.Tensor, lengths: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        heads = []
        head_tags = []
        for energy, length in zip(batch_energy.detach().cpu(), lengths):
            scores, tag_ids = energy.max(dim=0)
            # Although we need to include the root node so that the MST includes it,
            # we do not want any word to be the parent of the root node.
            # Here, we enforce this by setting the scores for all word -> ROOT edges
            # edges to be 0.
            scores[0, :] = 0
            # Decode the heads. Because we modify the scores to prevent
            # adding in word -> ROOT edges, we need to find the labels ourselves.
            instance_heads, _ = decode_mst(scores.numpy(), length, has_labels=False)

            # Find the labels which correspond to the edges in the max spanning tree.
            instance_head_tags = []
            for child, parent in enumerate(instance_heads):
                instance_head_tags.append(tag_ids[parent, child].item())
            # We don't care what the head or tag is for the root token, but by default
            # it's not necesarily the same in the batched vs unbatched case, which is
            # annoying. Here we'll just set them to zero.
            instance_heads[0] = 0
            instance_head_tags[0] = 0
            heads.append(instance_heads)
            head_tags.append(instance_head_tags)
        return (
            torch.from_numpy(np.stack(heads)).to(batch_energy.device),
            torch.from_numpy(np.stack(head_tags)).to(batch_energy.device),
        )

    def _get_head_tags(
        self,
        head_tag: torch.Tensor,
        child_tag: torch.Tensor,
        head_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decodes the head tags given the head and child tag representations
        and a tensor of head indices to compute tags for. Note that these are
        either gold or predicted heads, depending on whether this function is
        being called to compute the loss, or if it's being called during inference.

        # Parameters

        head_tag : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, tag_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag : `torch.Tensor`, required
            A tensor of shape (batch_size, sequence_length, tag_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        head_indices : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length). The indices of the heads
            for every word.

        # Returns

        head_tag_logits : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length, num_head_tags),
            representing logits for predicting a distribution over tags
            for each arc.
        """
        batch_size = head_tag.size(0)
        # shape (batch_size, 1)
        range_vector = torch.arange(batch_size, device=head_tag.device).unsqueeze(1)

        # This next statement is quite a complex piece of indexing, which you really need
        # to read the docs to understand. See here:
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#advanced-indexing
        # In effect, we are selecting the indices corresponding to the heads of each word
        # from the sequence length dimension for each element in the batch.

        # shape (batch_size, sequence_length, tag_dim)
        selected_head_tag = head_tag[range_vector, head_indices]
        selected_head_tag = selected_head_tag.contiguous()
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self.tag_bilinear(selected_head_tag, child_tag)
        return head_tag_logits

    def _get_mask_for_eval(
        self, mask: torch.BoolTensor, pos_tags: torch.LongTensor
    ) -> torch.Tensor:
        """
        Dependency evaluation excludes words are punctuation.
        Here, we create a new mask to exclude word indices which
        have a "punctuation-like" part of speech tag.

        # Parameters

        mask : `torch.BoolTensor`, required.
            The original mask.
        pos_tags : `torch.LongTensor`, required.
            The pos tags for the sequence.

        # Returns

        A new mask, where any indices equal to labels
        we should be ignoring are masked.
        """
        new_mask = mask.detach()
        for label in self._pos_to_ignore:
            label_mask = pos_tags.eq(label)
            new_mask = new_mask & ~label_mask
        return new_mask.bool()

    def training_step(self, batch, batch_idx):
        result = {}
        loss, _, _, _ = self.forward(batch)
        result["loss"] = loss
        return {
            "loss": result["loss"],
            "log": result,
        }

    def eval_helper(self, batch, prefix):
        loss, head_tag, child_tag, score_arc = self.forward(batch)
        lengths = self.get_mask(batch["sent"]).long().sum(dim=1).cpu().numpy()
        predicted_heads, predicted_labels = self._mst_decode(
            head_tag, child_tag, score_arc, batch["first_subword_mask"], lengths
        )
        evaluation_mask = self._get_mask_for_eval(
            batch["first_subword_mask"], batch["pos_tags"]
        )
        # ignore ROOT evaluation by default as ROOT token is not first subword
        assert (
            len(set(batch["lang"])) == 1
        ), "eval batch should contain only one language"
        lang = batch["lang"][0]

        self.metrics[lang].add(
            batch["heads"],
            batch["labels"],
            predicted_heads,
            predicted_labels,
            evaluation_mask,
        )

        result = dict()
        result[f"{prefix}_{lang}_loss"] = loss
        return result

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.eval_helper(batch, "val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.eval_helper(batch, "tst")

    def prepare_datasets(self, split: str) -> List[Dataset]:
        hparams = self.hparams
        data_class: Type[Dataset]
        if self.hparams.task == Task.parsing:
            data_class = ParsingDataset
        else:
            raise ValueError(f"Unsupported task: {hparams.task}")

        if split == Split.train:
            return self.prepare_datasets_helper(
                data_class,
                hparams.trn_langs,
                Split.train,
                hparams.max_trn_len,
                max_len_unit="subword",
            )
        elif split == Split.dev:
            return self.prepare_datasets_helper(
                data_class,
                hparams.val_langs,
                Split.dev,
                hparams.max_tst_len,
                max_len_unit="word",
            )
        elif split == Split.test:
            return self.prepare_datasets_helper(
                data_class,
                hparams.tst_langs,
                Split.test,
                hparams.max_tst_len,
                max_len_unit="word",
            )
        else:
            raise ValueError(f"Unsupported split: {hparams.split}")

    @classmethod
    def add_model_specific_args(cls, parser):
        parser.add_argument("--parser_use_pos", default=False, type=str2bool)
        parser.add_argument("--parser_predict_pos", default=False, type=str2bool)
        parser.add_argument("--parser_use_predict_pos", default=False, type=str2bool)
        parser.add_argument("--parser_pos_dim", default=100, type=int)
        parser.add_argument("--parser_tag_dim", default=128, type=int)
        parser.add_argument("--parser_arc_dim", default=512, type=int)
        parser.add_argument("--parser_dropout", default=0.33, type=float)
        return parser
