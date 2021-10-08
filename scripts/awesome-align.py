# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Modifications copyright (C) 2020 Zi-Yi Dou
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import itertools
import os
import random

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm, trange
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    BertForMaskedLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaForMaskedLM,
)


def return_extended_attention_mask(attention_mask, dtype):
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError("Wrong shape for input_ids or attention_mask")
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


class GuideHead(nn.Module):
    def __init__(self, pad_id, cls_id, sep_id):
        super().__init__()
        self.pad_id = pad_id
        self.cls_id = cls_id
        self.sep_id = sep_id

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (1, x.size(-1))
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states_src,
        hidden_states_tgt,
        inputs_src,
        inputs_tgt,
        guide=None,
        extraction="softmax",
        softmax_threshold=0.001,
        train_so=True,
        train_co=False,
        output_prob=False,
    ):
        # mask
        attention_mask_src = (
            (inputs_src == self.pad_id)
            + (inputs_src == self.cls_id)
            + (inputs_src == self.sep_id)
        ).float()
        attention_mask_tgt = (
            (inputs_tgt == self.pad_id)
            + (inputs_tgt == self.cls_id)
            + (inputs_tgt == self.sep_id)
        ).float()
        len_src = torch.sum(1 - attention_mask_src, -1)
        len_tgt = torch.sum(1 - attention_mask_tgt, -1)
        attention_mask_src = return_extended_attention_mask(
            1 - attention_mask_src, hidden_states_src.dtype
        )
        attention_mask_tgt = return_extended_attention_mask(
            1 - attention_mask_tgt, hidden_states_src.dtype
        )

        # qkv
        query_src = self.transpose_for_scores(hidden_states_src)
        query_tgt = self.transpose_for_scores(hidden_states_tgt)
        key_tgt = query_tgt

        # att
        attention_scores = torch.matmul(query_src, key_tgt.transpose(-1, -2))
        attention_scores_src = attention_scores + attention_mask_tgt
        attention_scores_tgt = attention_scores + attention_mask_src.transpose(-1, -2)

        assert extraction == "softmax"
        attention_probs_src = (
            nn.Softmax(dim=-1)(attention_scores_src)
            if extraction == "softmax"
            else None
        )
        attention_probs_tgt = (
            nn.Softmax(dim=-2)(attention_scores_tgt)
            if extraction == "softmax"
            else None
        )

        if guide is None:
            threshold = softmax_threshold if extraction == "softmax" else 0
            align_matrix = (attention_probs_src > threshold) * (
                attention_probs_tgt > threshold
            )
            if not output_prob:
                return align_matrix
            # A heuristic of generating the alignment probability
            attention_probs_src = nn.Softmax(dim=-1)(
                attention_scores_src / torch.sqrt(len_tgt.view(-1, 1, 1, 1))
            )
            attention_probs_tgt = nn.Softmax(dim=-2)(
                attention_scores_tgt / torch.sqrt(len_src.view(-1, 1, 1, 1))
            )
            align_prob = (2 * attention_probs_src * attention_probs_tgt) / (
                attention_probs_src + attention_probs_tgt + 1e-9
            )
            return align_matrix, align_prob

        so_loss = 0
        if train_so:
            so_loss_src = torch.sum(
                torch.sum(attention_probs_src * guide, -1), -1
            ).view(-1)
            so_loss_tgt = torch.sum(
                torch.sum(attention_probs_tgt * guide, -1), -1
            ).view(-1)

            so_loss = so_loss_src / len_src + so_loss_tgt / len_tgt
            so_loss = -torch.mean(so_loss)

        co_loss = 0
        if train_co:
            min_len = torch.min(len_src, len_tgt)
            trace = torch.matmul(
                attention_probs_src, (attention_probs_tgt).transpose(-1, -2)
            ).squeeze(1)
            trace = torch.einsum("bii->b", trace)
            co_loss = -torch.mean(trace / min_len)

        return so_loss + co_loss


class Aligner(nn.Module):
    def __init__(self, tokenizer, model):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model

        if isinstance(model, BertForMaskedLM):
            self.encoder = model.bert
            self.lm_head = model.cls
            self.hidden_size = model.config.hidden_size
            self.vocab_size = model.config.vocab_size
        elif isinstance(model, RobertaForMaskedLM):
            self.encoder = model.roberta
            self.lm_head = model.lm_head
            self.hidden_size = model.config.hidden_size
            self.vocab_size = model.config.vocab_size
        else:
            raise ValueError("Unsupported model:", model)

        self.guide_layer = GuideHead(
            self.tokenizer.pad_token_id,
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
        )

    def get_aligned_word(
        self,
        inputs_src,
        inputs_tgt,
        bpe2word_map_src,
        bpe2word_map_tgt,
        device,
        src_len,
        tgt_len,
        align_layer=8,
        extraction="softmax",
        softmax_threshold=0.001,
        test=False,
        output_prob=False,
        word_aligns=None,
    ):
        batch_size = inputs_src.size(0)
        bpelen_src, bpelen_tgt = inputs_src.size(1) - 2, inputs_tgt.size(1) - 2
        if word_aligns is None:
            inputs_src = inputs_src.to(dtype=torch.long, device=device).clone()
            inputs_tgt = inputs_tgt.to(dtype=torch.long, device=device).clone()

            with torch.no_grad():
                outputs_src = self.encoder(
                    inputs_src,
                    attention_mask=(inputs_src != self.tokenizer.pad_token_id),
                    output_hidden_states=True,
                ).hidden_states[align_layer]
                outputs_tgt = self.encoder(
                    inputs_tgt,
                    attention_mask=(inputs_tgt != self.tokenizer.pad_token_id),
                    output_hidden_states=True,
                ).hidden_states[align_layer]

                attention_probs_inter = self.guide_layer(
                    outputs_src,
                    outputs_tgt,
                    inputs_src,
                    inputs_tgt,
                    extraction=extraction,
                    softmax_threshold=softmax_threshold,
                    output_prob=output_prob,
                )
                if output_prob:
                    attention_probs_inter, alignment_probs = attention_probs_inter
                    alignment_probs = alignment_probs[:, 0, 1:-1, 1:-1]
                attention_probs_inter = attention_probs_inter.float()

            word_aligns = []
            attention_probs_inter = attention_probs_inter[:, 0, 1:-1, 1:-1]

            for idx, (attention, b2w_src, b2w_tgt) in enumerate(
                zip(attention_probs_inter, bpe2word_map_src, bpe2word_map_tgt)
            ):
                aligns = set() if not output_prob else dict()
                non_zeros = torch.nonzero(attention)
                for i, j in non_zeros:
                    word_pair = (b2w_src[i], b2w_tgt[j])
                    if output_prob:
                        prob = alignment_probs[idx, i, j]
                        if word_pair not in aligns:
                            aligns[word_pair] = prob
                        else:
                            aligns[word_pair] = max(aligns[word_pair], prob)
                    else:
                        aligns.add(word_pair)
                word_aligns.append(aligns)

        if test:
            return word_aligns

        guide = torch.zeros(batch_size, 1, src_len, tgt_len)
        for idx, (word_align, b2w_src, b2w_tgt) in enumerate(
            zip(word_aligns, bpe2word_map_src, bpe2word_map_tgt)
        ):
            len_src = min(bpelen_src, len(b2w_src))
            len_tgt = min(bpelen_tgt, len(b2w_tgt))
            for i in range(len_src):
                for j in range(len_tgt):
                    if (b2w_src[i], b2w_tgt[j]) in word_align:
                        guide[idx, 0, i + 1, j + 1] = 1.0
        return guide


def set_seed(args):
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path):
        assert os.path.isfile(file_path)
        print("Loading the dataset...")
        self.examples = []
        with open(file_path, encoding="utf-8") as f:
            for idx, line in enumerate(tqdm(f.readlines())):
                if (
                    len(line) == 0
                    or line.isspace()
                    or not len(line.split(" ||| ")) == 2
                ):
                    raise ValueError(f"Line {idx+1} is not in the correct format!")

                src, tgt = line.split(" ||| ")
                if src.rstrip() == "" or tgt.rstrip() == "":
                    raise ValueError(f"Line {idx+1} is not in the correct format!")

                sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
                token_src, token_tgt = [
                    tokenizer.tokenize(word) for word in sent_src
                ], [tokenizer.tokenize(word) for word in sent_tgt]
                wid_src, wid_tgt = [
                    tokenizer.convert_tokens_to_ids(x) for x in token_src
                ], [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]

                max_len = tokenizer.max_len_single_sentence
                if args.max_len != -1:
                    max_len = min(max_len, args.max_len)
                ids_src = tokenizer.prepare_for_model(
                    list(itertools.chain(*wid_src)),
                    return_tensors="pt",
                    max_length=max_len,
                    truncation=True,
                )["input_ids"]
                ids_tgt = tokenizer.prepare_for_model(
                    list(itertools.chain(*wid_tgt)),
                    return_tensors="pt",
                    max_length=max_len,
                    truncation=True,
                )["input_ids"]
                if len(ids_src) == 2 or len(ids_tgt) == 2:
                    raise ValueError(f"Line {idx+1} is not in the correct format!")

                bpe2word_map_src = []
                for i, word_list in enumerate(token_src):
                    bpe2word_map_src += [i for x in word_list]
                bpe2word_map_tgt = []
                for i, word_list in enumerate(token_tgt):
                    bpe2word_map_tgt += [i for x in word_list]

                self.examples.append(
                    (ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt)
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def word_align(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    def collate(examples):
        ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt = zip(*examples)
        ids_src = pad_sequence(
            ids_src, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        ids_tgt = pad_sequence(
            ids_tgt, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        return ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt

    dataset = LineByLineTextDataset(tokenizer, args, file_path=args.data_file)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.batch_size, collate_fn=collate
    )

    model.to(args.device)
    model.eval()
    aligner = Aligner(tokenizer, model)
    tqdm_iterator = trange(dataset.__len__(), desc="Extracting")
    if args.output_prob_file is not None:
        prob_writer = open(args.output_prob_file, "w")
    with open(args.output_file, "w") as writer:
        for batch in dataloader:
            with torch.no_grad():
                ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt = batch
                word_aligns_list = aligner.get_aligned_word(
                    ids_src,
                    ids_tgt,
                    bpe2word_map_src,
                    bpe2word_map_tgt,
                    args.device,
                    0,
                    0,
                    align_layer=args.align_layer,
                    extraction=args.extraction,
                    softmax_threshold=args.softmax_threshold,
                    test=True,
                    output_prob=(args.output_prob_file is not None),
                )
                for word_aligns in word_aligns_list:
                    output_str = []
                    if args.output_prob_file is not None:
                        output_prob_str = []
                    for word_align in word_aligns:
                        output_str.append(f"{word_align[0]}-{word_align[1]}")
                        if args.output_prob_file is not None:
                            output_prob_str.append(f"{word_aligns[word_align]}")
                    writer.write(" ".join(output_str) + "\n")
                    if args.output_prob_file is not None:
                        prob_writer.write(" ".join(output_prob_str) + "\n")
                tqdm_iterator.update(len(ids_src))
    if args.output_prob_file is not None:
        prob_writer.close()


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_file",
        default=None,
        type=str,
        required=True,
        help="The input data file (a text file).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="The output file.",
    )
    parser.add_argument(
        "--align_layer", type=int, default=8, help="layer for alignment extraction"
    )
    parser.add_argument(
        "--extraction", default="softmax", type=str, help="softmax or entmax15"
    )
    parser.add_argument("--softmax_threshold", type=float, default=0.001)
    parser.add_argument("--max_len", type=int, default=-1)
    parser.add_argument(
        "--output_prob_file",
        default=None,
        type=str,
        help="The output probability file.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    args = parser.parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    args.device = device

    # Set seed
    set_seed(args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, cache_dir=args.cache_dir
    )
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_name_or_path, cache_dir=args.cache_dir
    )

    word_align(args, model, tokenizer)


if __name__ == "__main__":
    main()
