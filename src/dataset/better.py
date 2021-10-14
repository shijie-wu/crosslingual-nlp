import json
import os
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple

from sacremoses import MosesTokenizer

from dataset.base import Dataset


class BetterDataset(Dataset):
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

    @classmethod
    def read_file(cls, filepath: str, lang: str, split: str) -> Iterator[Dict]:
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Could not find {filepath}")
        mt = MosesTokenizer(lang="en")
        with open(filepath, "r") as f:
            org_data = json.load(f)
        entries = org_data["entries"]
        for entry_k, entry in entries.items():
            segment_text = "TMPLASTWORD " + entry["segment-text"] + " TMPLASTWORD"
            segment_text_tok = mt.tokenize(segment_text, escape=False)
            # segments.append(segment_text_tok[1:-1])
            annotation_sets = entry["annotation-sets"]

            events = annotation_sets["abstract-events"]
            span_sets = events["span-sets"]
            strings = []
            hstrings = []
            for span_set_k, span_set in span_sets.items():
                spans = span_set["spans"]
                for span in spans:
                    string = "TMPLASTWORD " + span["string"] + " TMPLASTWORD"
                    string_tok = mt.tokenize(string, escape=False)
                    strings.append(string_tok[1:-1])
                    if "hstring" in span.keys():
                        hstring = "TMPLASTWORD " + span["hstring"] + " TMPLASTWORD"
                        hstring_tok = mt.tokenize(hstring, escape=False)
                        hstrings.append(hstring_tok[1:-1])
                    else:
                        hstrings.append([])

            yield {
                "sent": segment_text_tok[1:-1],
                "span_strings": strings,
                "span_hstrings": hstrings,
            }

    @classmethod
    def project_label(
        cls, example: Dict, translation: List[str], mapping: List[Tuple]
    ) -> Dict:
        # span projection
        src2tgt = defaultdict(list)
        for src_idx, tgt_idx in mapping:
            src2tgt[src_idx + 1].append(tgt_idx + 1)

        span_strings_tgt, span_offsets_tgt, span_hstrings_tgt, span_hoffsets_tgt = (
            [],
            [],
            [],
            [],
        )
        for ii, ss in enumerate(example["span_strings"]):
            found = cls.match_span_segment(example["sent"], ss)
            ts, (start, end) = cls.project_label_util(src2tgt, translation, found)
            span_strings_tgt.append(ts)
            span_offsets_tgt.append((start, end))

        for ii, ss in enumerate(example["span_hstrings"]):
            if ss:
                found = cls.match_span_segment(example["sent"], ss)
                ts, (start, end) = cls.project_label_util(src2tgt, translation, found)
                span_hstrings_tgt.append(ts)
                span_hoffsets_tgt.append((start, end))
            else:
                span_hstrings_tgt.append([])
                span_hoffsets_tgt.append((-1, -1))

        return {
            "spans_tgt": span_strings_tgt,
            "offsets_tgt": span_offsets_tgt,
            "hspans_tgt": span_hstrings_tgt,
            "hoffsets_tgt": span_hoffsets_tgt,
            "translation": translation,
        }

    @classmethod
    def write_example(cls, example: Dict, file_handler):

        with open(example["file_path"][0], "r") as f:
            org_data = json.load(f)

        entries = org_data["entries"]
        i = 0
        for entry_k, entry in entries.items():
            # update segment text with its tgt translation
            entry["segment-text"] = " ".join(example["translation"][i])
            annotation_sets = entry["annotation-sets"]
            events = annotation_sets["abstract-events"]
            span_sets = events["span-sets"]
            s = 0
            for span_set_k, span_set in span_sets.items():
                spans = span_set["spans"]
                for span in spans:
                    # update the span string with the string in tgt lang
                    span["string"] = " ".join(example["spans_tgt"][i][s])
                    if "hstring" in span.keys():
                        # update the span head string with the string in tgt lang
                        span["hstring"] = " ".join(example["hspans_tgt"][i][s])
                    s = s + 1
            i = i + 1

            if "segment-text-tok" in entry:
                del entry["segment-text-tok"]
            if "char2tok" in entry:
                del entry["char2tok"]
            if "tok2char" in entry:
                del entry["tok2char"]

        # write to a new json file

        json.dump(org_data, file_handler, indent=2, ensure_ascii=False)

    @classmethod
    def get_file(cls, path: str, lang: str, split: str) -> Optional[str]:
        if split == "train":
            return f"{path}/train.tok"
        elif split == "analysis":
            return f"{path}/analysis.tok"
        elif split == "devtest":
            return f"{path}/devtest.tok"
        else:
            raise ValueError(f"Unsupported split: {split}")
