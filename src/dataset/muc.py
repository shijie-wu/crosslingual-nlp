import json
import os
from collections import OrderedDict, defaultdict
from typing import Dict, Iterator, List, Optional, Tuple

from nltk.tokenize import sent_tokenize as nltk_sent_tokenize
from sacremoses import MosesTokenizer

from dataset.base import Dataset


class MUCDataset(Dataset):
    @staticmethod
    def match_span_segment(seg, string):
        cursor = 0
        found = []
        for i, s in enumerate(seg):
            if s == string[cursor]:
                cursor += 1
                if cursor == len(string):
                    for x in reversed(range(len(string))):
                        found.append(i - x + 1)
                    # cursor = 0
                    break  # find only the first occurence
            else:
                cursor = 0
        # if (len(found) == 0):
        #  print("NOT FOUND", string, seg)
        return found

    @staticmethod
    def project_label_util(maps, translation, findings):
        tgt_span_list = []  # list of all tgt words for a span string
        for fo in findings:
            if fo in maps:
                for item in maps[fo]:
                    tgt_span_list.append(item)
            # else: # fix @-@ in tokenization
            #  print("tgt word not found", segment)
        tgt_span_list.sort()
        tgt_span = []
        start = 0
        end = 0
        if tgt_span_list:
            # read all words in between
            leng = 0
            for ts in range(tgt_span_list[0], tgt_span_list[-1] + 1):
                tgt_span.append(translation[ts - 1])
                leng += len(translation[ts - 1]) + 1
            leng -= 1
            for ll in range(tgt_span_list[0]):
                if ll > 0:
                    start += len(translation[ll - 1]) + 1
            end = start + leng

        return tgt_span, (start, end)

    @staticmethod
    def read_file_util(ls):
        out = []
        mt = MosesTokenizer(lang="en")
        if len(ls) > 0:
            for ann in ls:
                out.append([])
                for i in ann:
                    string = "TMPLASTWORD " + i + " TMPLASTWORD"
                    string_tok = mt.tokenize(string, escape=False)
                    out[-1].append(string_tok[1:-1])
        return out

    @classmethod
    def read_file(cls, filepath: str, lang: str, split: str) -> Iterator[Dict]:
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Could not find {filepath}")
        mt = MosesTokenizer(lang="en")
        with open(filepath, "r") as f:
            org_data = json.load(f)
        for entry_k, entry in org_data.items():
            sentences = []
            doc_text = "TMPLASTWORD " + entry["doc"] + " TMPLASTWORD"
            doc_text_tok = mt.tokenize(doc_text, escape=False)
            temp = entry["doc"]
            sentence_splits = nltk_sent_tokenize(temp)
            for jj in sentence_splits:
                segment_text = "TMPLASTWORD " + jj + " TMPLASTWORD"
                segment_text_tok = mt.tokenize(segment_text, escape=False)
                sentences.append(segment_text_tok[1:-1])
            # segments.append(segment_text_tok[1:-1])
            annotation_sets = entry["roles"]

            perp_individual_id = annotation_sets["perp_individual_id"]
            perp_organization_id = annotation_sets["perp_organization_id"]
            phys_tgt_id = annotation_sets["phys_tgt_id"]
            hum_tgt_name = annotation_sets["hum_tgt_name"]
            incident_instrument_id = annotation_sets["incident_instrument_id"]

            perp_individual_id = cls.read_file_util(perp_individual_id)
            perp_organization_id = cls.read_file_util(perp_organization_id)
            phys_tgt_id = cls.read_file_util(phys_tgt_id)
            hum_tgt_name = cls.read_file_util(hum_tgt_name)
            incident_instrument_id = cls.read_file_util(incident_instrument_id)

            yield {
                "doc": doc_text_tok[1:-1],
                "sent": sentences,
                "perp_individual_id": perp_individual_id,
                "perp_organization_id": perp_organization_id,
                "phys_tgt_id": phys_tgt_id,
                "hum_tgt_name": hum_tgt_name,
                "incident_instrument_id": incident_instrument_id,
                "id": entry_k,
            }

    @classmethod
    def project_label(
        cls, example: Dict, translation: List[str], mapping: List[Tuple]
    ) -> Dict:
        # span projection
        src2tgt = defaultdict(list)
        for src_idx, tgt_idx in mapping:
            src2tgt[src_idx + 1].append(tgt_idx + 1)

        perp_individual_id_tgt: List[list] = []
        perp_organization_id_tgt: List[list] = []
        phys_tgt_id_tgt: List[list] = []
        hum_tgt_name_tgt: List[list] = []
        incident_instrument_id_tgt: List[list] = []

        for ann in example["perp_individual_id"]:
            perp_individual_id_tgt.append([])
            for ii, ss in enumerate(ann):
                found = cls.match_span_segment(example["doc"], ss)
                ts, (start, end) = cls.project_label_util(src2tgt, translation, found)
                perp_individual_id_tgt[-1].append(" ".join(ts))

        for ann in example["perp_organization_id"]:
            perp_organization_id_tgt.append([])
            for ii, ss in enumerate(ann):
                found = cls.match_span_segment(example["doc"], ss)
                ts, (start, end) = cls.project_label_util(src2tgt, translation, found)
                perp_organization_id_tgt[-1].append(" ".join(ts))

        for ann in example["phys_tgt_id"]:
            phys_tgt_id_tgt.append([])
            for ii, ss in enumerate(ann):
                found = cls.match_span_segment(example["doc"], ss)
                ts, (start, end) = cls.project_label_util(src2tgt, translation, found)
                phys_tgt_id_tgt[-1].append(" ".join(ts))

        for ann in example["hum_tgt_name"]:
            hum_tgt_name_tgt.append([])
            for ii, ss in enumerate(ann):
                found = cls.match_span_segment(example["doc"], ss)
                ts, (start, end) = cls.project_label_util(src2tgt, translation, found)
                hum_tgt_name_tgt[-1].append(" ".join(ts))

        for ann in example["incident_instrument_id"]:
            incident_instrument_id_tgt.append([])
            for ii, ss in enumerate(ann):
                found = cls.match_span_segment(example["doc"], ss)
                ts, (start, end) = cls.project_label_util(src2tgt, translation, found)
                incident_instrument_id_tgt[-1].append(" ".join(ts))

        roles = OrderedDict(
            {
                "perp_individual_id": perp_individual_id_tgt,
                "perp_organization_id": perp_organization_id_tgt,
                "phys_tgt_id": phys_tgt_id_tgt,
                "hum_tgt_name": hum_tgt_name_tgt,
                "incident_instrument_id": incident_instrument_id_tgt,
            }
        )
        return OrderedDict(
            {example["id"]: {"doc": " ".join(translation), "roles": roles}}
        )

    @classmethod
    def write_example(cls, example: Dict, file_handler):
        # write to a new json file
        json.dump(example, file_handler, indent=2, ensure_ascii=False)

    @classmethod
    def get_file(cls, path: str, lang: str, split: str) -> Optional[str]:
        if split == "train":
            return f"{path}/train_full.json"
        elif split == "dev":
            return f"{path}/dev_full.json"
        elif split == "test":
            return f"{path}/test.json"
        else:
            raise ValueError(f"Unsupported split: {split}")
