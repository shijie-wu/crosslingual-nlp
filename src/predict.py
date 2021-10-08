import glob
from functools import partial

import numpy as np
from fire import Fire
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset import ParsingDataset, UdPOS, WikiAnnNER
from model import DependencyParser, Tagger
from util import default_collate


def read_file(filepath, tknzr, lang, max_len=500):
    with open(filepath) as fp:
        for line in tqdm(fp.readlines()):
            line = line.strip()
            if not line:
                continue
            orig_sent = []
            sent = []
            first_subword_mask = []
            for word in line.split():
                subwords = tknzr.tokenize(word)
                if len(sent) + len(subwords) > max_len:
                    break
                if subwords:
                    orig_sent.append(word)
                for i, w in enumerate(subwords):
                    sent.append(w)
                    first_subword_mask.append(1 if i == 0 else 0)
            sent = tknzr.convert_tokens_to_ids(sent)
            sent = tknzr.build_inputs_with_special_tokens(sent)
            first_subword_mask = tknzr.build_inputs_with_special_tokens(
                first_subword_mask
            )
            first_subword_mask[0] = 0
            first_subword_mask[-1] = 0
            sent = np.array(sent)
            first_subword_mask = np.array(first_subword_mask)
            assert len(orig_sent) == sum(first_subword_mask)
            assert len(sent) == len(first_subword_mask)
            yield {
                "sent": sent,
                "first_subword_mask": first_subword_mask,
                "orig_sent": orig_sent,
                "lang": lang,
            }


def get_model_path(filepath):
    model_path = glob.glob(filepath, recursive=True)
    assert len(model_path) == 1, model_path
    return model_path[0]


def main(
    filepath,
    encoder,
    lang,
    output_file,
    task,
    parser_path=None,
    tagger_path=None,
    batch_size=16,
):
    assert task in ["ner", "ud", "udpos"]
    if task == "ner":
        assert parser_path is None and tagger_path is not None
    elif task == "ud":
        assert parser_path is not None and tagger_path is not None
    elif task == "udpos":
        assert parser_path is None and tagger_path is not None

    tknzr = AutoTokenizer.from_pretrained(encoder)
    examples = list(read_file(filepath, tknzr, lang))

    if task == "ner":
        CLASS = WikiAnnNER
        tagger = Tagger.load_from_checkpoint(get_model_path(tagger_path))
        tagger = tagger.to("cuda")
        tagger.eval()

    elif task == "ud":
        CLASS = ParsingDataset
        parser = DependencyParser.load_from_checkpoint(get_model_path(parser_path))
        parser = parser.to("cuda")
        parser.eval()
        tagger = Tagger.load_from_checkpoint(get_model_path(tagger_path))
        tagger = tagger.to("cuda")
        tagger.eval()

    elif task == "udpos":
        CLASS = UdPOS
        tagger = Tagger.load_from_checkpoint(get_model_path(tagger_path))
        tagger = tagger.to("cuda")
        tagger.eval()

    padding = {
        "sent": tknzr.pad_token_id,
        "first_subword_mask": 0,
        "orig_sent": 0,
        "lang": 0,
    }
    collate = partial(default_collate, padding=padding)
    dataloader = DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate,
    )

    with open(output_file, "w") as fp:
        for batch in tqdm(dataloader):
            batch["sent"] = batch["sent"].to("cuda")
            batch["first_subword_mask"] = batch["first_subword_mask"].to("cuda")
            if task == "ner" or task == "udpos":
                tagger_output = tagger.predict(batch)
                for i in range(len(tagger_output)):
                    CLASS.write_example(tagger_output[i], fp)

            elif task == "ud":
                parser_output = parser.predict(batch)
                tagger_output = tagger.predict(batch)
                for i in range(len(parser_output)):
                    assert parser_output[i]["sent"] == tagger_output[i]["sent"]
                    parser_output[i]["pos_tags"] = tagger_output[i]["labels"]
                    CLASS.write_example(parser_output[i], fp)


if __name__ == "__main__":
    Fire(main)
