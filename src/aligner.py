from collections import defaultdict
from copy import deepcopy
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import util
from base_model import Model
from dataset import Bitext, Dataset
from enumeration import Split
from metric import AccuracyMetric

PAD_ALIGN = -1


class Aligner(Model):
    def __init__(self, hparams):
        super(Aligner, self).__init__(hparams)

        self._comparsion = "min"
        self._selection_criterion = "val_loss"

        self.aligner_projector = self.build_aligner_projector()
        self.orig_model = deepcopy(self.model)
        util.freeze(self.orig_model)

        self.mappings = nn.ModuleList([])
        for _ in range(self.num_layers):
            m = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            nn.init.eye_(m.weight)
            self.mappings.append(m)

        self.padding = {
            "src_sent": self.tokenizer.pad_token_id,
            "tgt_sent": self.tokenizer.pad_token_id,
            "src_align": PAD_ALIGN,
            "tgt_align": PAD_ALIGN,
            "src_lang": 0,
            "tgt_lang": 0,
            "lang": 0,
        }

        self.setup_metrics()

    def setup_metrics(self):
        langs = self.hparams.trn_langs + self.hparams.val_langs + self.hparams.tst_langs
        langs = sorted(list(set(langs)))
        for lang in langs:
            if self.hparams.aligner_sim in ["linear", "l2"]:
                self.metrics = {}
                return
            elif self.hparams.aligner_sim == "cntrstv":
                self.metrics[f"{lang}-fwd"] = AccuracyMetric()
                self.metrics[f"{lang}-bwd"] = AccuracyMetric()
            elif self.hparams.aligner_sim == "jnt_cntrstv":
                self.metrics[f"{lang}-joint"] = AccuracyMetric()
            else:
                raise ValueError(self.hparams.aligner_sim)
        self.reset_metrics()

    def aggregate_metrics(self, langs: List[str], prefix: str):
        if self.hparams.aligner_sim in ["linear", "l2"]:
            langs = []
        elif self.hparams.aligner_sim == "cntrstv":
            langs = sum([[f"{lng}-fwd", f"{lng}-bwd"] for lng in langs], [])
        elif self.hparams.aligner_sim == "jnt_cntrstv":
            langs = [f"{lng}-joint" for lng in langs]
        else:
            raise ValueError(self.hparams.aligner_sim)

        aver_metric = defaultdict(list)
        for lang in langs:
            metric = self.metrics[lang]
            for key, val in metric.get_metric().items():
                self.log(f"{prefix}_{lang}_{key}", val)

                aver_metric[key].append(val)

        for key, vals in aver_metric.items():
            self.log(f"{prefix}_{key}", torch.stack(vals).mean())

    def build_aligner_projector(self):
        if self.hparams.aligner_projector == "id":
            return nn.Identity()
        elif self.hparams.aligner_projector == "linear":
            return nn.Linear(
                self.hidden_size, self.hparams.aligner_proj_dim, bias=False
            )
        elif self.hparams.aligner_projector == "nn":
            return nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size, bias=False),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hparams.aligner_proj_dim, bias=False),
            )
        else:
            raise ValueError(self.hparams.aligner_projector)

    def preprocess_batch(self, batch):
        def helper(sent, align):
            bs, seq_len = sent.shape
            shift = torch.arange(bs, device=sent.device).view(-1, 1) * seq_len
            index = shift.masked_fill(align == PAD_ALIGN, 0) + align
            return index[index != PAD_ALIGN]

        batch["src_align"] = helper(batch["src_sent"], batch["src_align"])
        batch["tgt_align"] = helper(batch["tgt_sent"], batch["tgt_align"])

        assert batch["src_align"].shape == batch["tgt_align"].shape
        seq_len = batch["src_align"].shape[0]
        device = batch["src_align"].device
        if self.hparams.aligner_sim == "cntrstv":
            batch["goal"] = torch.arange(seq_len, device=device)
        elif self.hparams.aligner_sim == "jnt_cntrstv":
            goal = torch.arange(seq_len * 2, device=device)
            i, j = torch.split(goal, len(goal) // 2)
            batch["goal"] = torch.cat((j, i))

        return batch

    def l2_loss(self, src_hid, tgt_hid):
        loss = torch.norm(src_hid - tgt_hid, dim=1).square().mean()
        return loss, None

    def linear_loss(self, src_hids, tgt_hids):
        if self.hparams.aligner_orthogonal > 0:
            for mapping in self.mappings:
                W = mapping.weight.data
                beta = self.hparams.aligner_orthogonal
                W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))

        loss = []
        for src, tgt, mapping in zip(src_hids, tgt_hids, self.mappings):
            l, _ = self.l2_loss(src, mapping(tgt))
            loss.append(l)
        return torch.mean(torch.stack(loss)), loss

    def cntrstv_loss(self, src_hid, tgt_hid, goal):
        tgt_hid = tgt_hid.transpose(0, 1)
        sim = torch.matmul(src_hid, tgt_hid)
        if self.hparams.aligner_normalize:
            sim = sim / src_hid.norm(dim=1, keepdim=True)
            sim = sim / tgt_hid.norm(dim=0, keepdim=True)
        sim = sim / self.hparams.aligner_temperature
        # given a src, pick a tgt from tgts
        fwd_logits = F.log_softmax(sim, dim=-1)
        # given a tgt, pick a src from srcs
        bwd_logits = F.log_softmax(sim.transpose(0, 1), dim=-1)

        loss = (F.nll_loss(fwd_logits, goal) + F.nll_loss(bwd_logits, goal)) / 2
        return loss, (fwd_logits, bwd_logits)

    def jnt_cntrstv_loss(self, src_hid, tgt_hid, goal):
        hid = torch.cat((src_hid, tgt_hid))
        seq_len = hid.shape[0]
        sim = torch.matmul(hid, hid.transpose(0, 1))
        if self.hparams.aligner_normalize:
            hid_norm = hid.norm(dim=1, keepdim=True)
            sim = sim / hid_norm
            sim = sim / hid_norm.transpose(0, 1)
        sim = sim / self.hparams.aligner_temperature
        sim = sim - torch.eye(seq_len, device=hid.device) * 1e6
        # given a src, pick a tgt from srcs & tgts
        joint_logits = F.log_softmax(sim, dim=-1)

        loss = F.nll_loss(joint_logits, goal)
        return loss, joint_logits

    def l2_param_reg(self):
        old_params = {k: v for k, v in self.orig_model.named_parameters()}
        new_params = {k: v for k, v in self.model.named_parameters()}
        assert old_params.keys() == new_params.keys()
        l2_loss = 0
        for k in sorted(old_params.keys()):
            l2_loss += torch.norm(old_params[k] - new_params[k]) ** 2
        return torch.sqrt(l2_loss)

    def l2_src_reg(self, src_sent, src_hs):
        self.orig_model.eval()
        orig_src_hs = self.encode_sent(src_sent, langs=None, model=self.orig_model)
        mask = self.get_mask(src_sent).unsqueeze(-1)
        return torch.norm((orig_src_hs - src_hs) * mask) ** 2 / mask.sum()

    def forward(self, batch, prefix):
        batch = self.preprocess_batch(batch)

        if self.hparams.aligner_sim == "linear":
            src_hs = self.encode_sent(
                batch["src_sent"], langs=None, return_raw_hidden_states=True
            )
            tgt_hs = self.encode_sent(
                batch["tgt_sent"], langs=None, return_raw_hidden_states=True
            )

            src_aligned_hs = [
                h.view(-1, self.hidden_size)[batch["src_align"]] for h in src_hs
            ]
            tgt_aligned_hs = [
                h.view(-1, self.hidden_size)[batch["tgt_align"]] for h in tgt_hs
            ]

        else:
            src_hs = self.encode_sent(batch["src_sent"], langs=None)
            tgt_hs = self.encode_sent(batch["tgt_sent"], langs=None)

            src_aligned_hs = src_hs.view(-1, self.hidden_size)[batch["src_align"]]
            tgt_aligned_hs = tgt_hs.view(-1, self.hidden_size)[batch["tgt_align"]]

            src_aligned_hs = self.aligner_projector(src_aligned_hs)
            tgt_aligned_hs = self.aligner_projector(tgt_aligned_hs)

        result = dict()
        if self.hparams.aligner_sim == "l2":
            loss, extra = self.l2_loss(src_aligned_hs, tgt_aligned_hs)
            result[f"{prefix}_l2_loss"] = loss
        elif self.hparams.aligner_sim == "linear":
            loss, extra = self.linear_loss(src_aligned_hs, tgt_aligned_hs)
            result[f"{prefix}_linear_loss"] = loss
            for i, layer_loss in enumerate(extra):
                result[f"{prefix}_layer{i}_linear_loss"] = layer_loss
        elif self.hparams.aligner_sim == "cntrstv":
            loss, extra = self.cntrstv_loss(
                src_aligned_hs, tgt_aligned_hs, batch["goal"]
            )
            result[f"{prefix}_cntrstv_loss"] = loss
            result[f"{prefix}_num_examples"] = src_aligned_hs.shape[0] - 1.0
        elif self.hparams.aligner_sim == "jnt_cntrstv":
            loss, extra = self.jnt_cntrstv_loss(
                src_aligned_hs, tgt_aligned_hs, batch["goal"]
            )
            result[f"{prefix}_jnt_cntrstv_loss"] = loss
            result[f"{prefix}_num_examples"] = src_aligned_hs.shape[0] * 2 - 1.0
        else:
            raise ValueError(self.hparams.aligner_sim)
        return result, src_hs, loss, extra

    def training_step(self, batch, batch_idx):
        result, src_hs, loss, extra = self.forward(batch, "trn")
        for key, val in result.items():
            self.log(key, val)

        l2_param_reg = 0.0
        l2_param_coeff = self.hparams.aligner_l2_param_coeff
        if l2_param_coeff > 0:
            l2_param_reg = self.l2_param_reg()
            self.log("l2_param_reg", l2_param_reg)

        l2_src_reg = 0.0
        l2_src_coeff = self.hparams.aligner_l2_src_coeff
        if l2_src_coeff > 0:
            l2_src_reg = self.l2_src_reg(batch["src_sent"], src_hs)
            self.log("l2_src_reg", l2_src_reg)

        loss = loss + l2_param_coeff * l2_param_reg + l2_src_coeff * l2_src_reg
        self.log("loss", loss)

        return loss

    def evaluation_step_helper(self, batch, prefix):
        res, _, loss, extra = self.forward(batch, prefix)

        assert (
            len(set(batch["lang"])) == 1
        ), "eval batch should contain only one language pairs"
        lang = batch["lang"][0]

        result = dict()
        result[f"{prefix}_{lang}_loss"] = loss

        if self.hparams.aligner_sim == "l2":
            pass
        elif self.hparams.aligner_sim == "linear":
            for i, layer_loss in enumerate(extra):
                result[f"{prefix}_{lang}_layer{i}_loss"] = layer_loss
        elif self.hparams.aligner_sim == "cntrstv":
            result[f"{prefix}_num_examples"] = torch.tensor(
                res[f"{prefix}_num_examples"]
            )
            fwd_logits, bwd_logits = extra
            self.metrics[f"{lang}-fwd"].add(batch["goal"], fwd_logits)
            self.metrics[f"{lang}-bwd"].add(batch["goal"], bwd_logits)
        elif self.hparams.aligner_sim == "jnt_cntrstv":
            result[f"{prefix}_num_examples"] = torch.tensor(
                res[f"{prefix}_num_examples"]
            )
            joint_logits = extra
            self.metrics[f"{lang}-joint"].add(batch["goal"], joint_logits)

        return result

    def prepare_datasets(self, split: str) -> List[Dataset]:
        hparams = self.hparams
        if split == Split.train:
            return self.prepare_datasets_helper(
                Bitext, hparams.trn_langs, Split.train, hparams.max_trn_len
            )
        elif split == Split.dev:
            return self.prepare_datasets_helper(
                Bitext, hparams.val_langs, Split.dev, hparams.max_tst_len
            )
        elif split == Split.test:
            return self.prepare_datasets_helper(
                Bitext, hparams.tst_langs, Split.test, hparams.max_tst_len
            )
        else:
            raise ValueError(f"Unsupported split: {hparams.split}")

    @classmethod
    def add_model_specific_args(cls, parser):
        # fmt: off
        parser.add_argument("--aligner_l2_param_coeff", default=0, type=float)
        parser.add_argument("--aligner_l2_src_coeff", default=0, type=float)
        parser.add_argument("--aligner_sim", choices=["linear", "l2", "cntrstv", "jnt_cntrstv"], type=str)
        parser.add_argument("--aligner_projector", default="id", choices=["id", "linear", "nn"], type=str)
        parser.add_argument("--aligner_proj_dim", default=128, type=int)
        parser.add_argument("--aligner_temperature", default=1.0, type=float)
        parser.add_argument("--aligner_normalize", default=False, type=util.str2bool)
        parser.add_argument("--aligner_orthogonal", default=0.0, type=float)
        # fmt: on
        return parser
