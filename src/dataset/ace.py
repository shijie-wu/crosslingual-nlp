import json
import os
from typing import Dict, Iterator, List, Optional, Tuple

import jsonlines
from nltk import wordpunct_tokenize as wordpunct_tokenize_
from nltk.tokenize.treebank import TreebankWordDetokenizer

import dataset.ace_util as ace
from dataset.base import Dataset


class ACEDataset(Dataset):
    @classmethod
    def read_file(cls, filepath: str, lang: str, split: str) -> Iterator[Dict]:
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Could not find {filepath}")

        with jsonlines.open(filepath) as reader:
            for doc_i, doc in enumerate(reader):
                doc_id = doc["doc_id"]
                sents = []
                for s in doc["sentences"]:
                    entities = []
                    for e in s["entities"]:
                        e_i = ace.Entity(
                            e["start"],
                            e["end"],
                            e["text"],
                            e["entity_id"],
                            e["mention_id"],
                            e["entity_type"],
                            e["entity_subtype"],
                            e["mention_type"],
                        )
                        entities.append(e_i)
                    relations = []
                    for r in s["relations"]:
                        arg1 = ace.RelationArgument(
                            r["arg1"]["mention_id"],
                            r["arg1"]["role"],
                            r["arg1"]["text"],
                        )
                        arg2 = ace.RelationArgument(
                            r["arg2"]["mention_id"],
                            r["arg2"]["role"],
                            r["arg2"]["text"],
                        )
                        r_i = ace.Relation(
                            r["relation_id"],
                            r["relation_type"],
                            r["relation_subtype"],
                            arg1,
                            arg2,
                        )
                        relations.append(r_i)
                    events = []
                    for e in s["events"]:
                        event_args = []
                        for a in e["arguments"]:
                            event_args.append(
                                ace.EventArgument(a["mention_id"], a["role"], a["text"])
                            )
                        e_e = ace.Event(
                            e["event_id"],
                            e["mention_id"],
                            e["event_type"],
                            e["event_subtype"],
                            ace.Span(
                                e["trigger"]["start"],
                                e["trigger"]["end"],
                                e["trigger"]["text"],
                            ),
                            event_args,
                        )
                        events.append(e_e)
                    sent_i = ace.Sentence(
                        start=s["start"],
                        end=s["end"],
                        text=s["text"],
                        sent_id=s["sent_id"],
                        tokens=s["tokens"],
                        entities=entities,
                        relations=relations,
                        events=events,
                    )
                    sents.append(sent_i)
                doc_ai = ace.Document(doc_id, sents)
                yield doc_ai.to_dict()

    @staticmethod
    def get_tgt_span(src_st, src_end, alignments):
        all_tgt = set()
        for j in range(src_st, src_end):
            if j <= max(alignments) and j in alignments:  # check why not true
                for k in alignments[j]:
                    all_tgt.add(k)
        if all_tgt:
            return min(all_tgt), max(all_tgt)
        else:
            return 0, 0

    @staticmethod
    def isSubArray(A, B):
        n = len(A)
        m = len(B)
        i = 0
        j = 0
        while i < n and j < m:
            if A[i] == B[j]:
                i += 1
                j += 1
                if j == m:
                    return i
            else:
                i = i - j + 1
                j = 0
        return -1

    @classmethod
    def update_arg(cls, stoks, arg, segments_tgt, alignments):
        arg = arg.replace("\n", " ")
        src_last_idx = cls.isSubArray(stoks, wordpunct_tokenize_(arg))
        if src_last_idx == -1:
            # fix later. 29 cases. Due to chunking sentence tokens based on entities and events texts.
            tgt = ""
        else:
            tgt_st, tgt_end = cls.get_tgt_span(
                src_last_idx - len(wordpunct_tokenize_(arg)) + 1,
                src_last_idx + 1,
                alignments,
            )
            tgt = ""
            if tgt_st > 0:
                for t in range(tgt_st - 1, tgt_end):
                    tgt += " " + segments_tgt[t]
        return tgt

    @classmethod
    def get_filter_ents(cls, entities, segments_tgt, alignments):
        filter_entities = []
        for e in entities:
            tgt_st, tgt_end = cls.get_tgt_span(e["start"] + 1, e["end"] + 1, alignments)
            tgt = ""
            if tgt_st > 0:
                for t in range(tgt_st - 1, tgt_end):
                    tgt += " " + segments_tgt[t]
            e["text"] = tgt.strip()
            if tgt_st > 0:
                e["start"] = tgt_st - 1
            else:
                e["start"] = 0
            e["end"] = tgt_end
            if e["text"] == "":
                filter_entities.append(e["mention_id"])
        return filter_entities

    @classmethod
    def get_filter_rels(cls, s, filter_entities, segments_tgt, alignments):
        filter_relation = []
        for r in s["relations"]:
            r["arg1"]["text"] = cls.update_arg(
                s["tokens"], r["arg1"]["text"], segments_tgt, alignments
            ).strip()
            r["arg2"]["text"] = cls.update_arg(
                s["tokens"], r["arg2"]["text"], segments_tgt, alignments
            ).strip()
            if (
                r["arg1"]["text"] != ""
                and r["arg2"]["text"] != ""
                and r["arg1"]["mention_id"] not in filter_entities
                and r["arg2"]["mention_id"] not in filter_entities
            ):
                pass
            else:
                filter_relation.append(r)
        return filter_relation

    @classmethod
    def get_filter_evts(cls, s, filter_entities, segments_tgt, alignments):
        orig_args = 0
        saved_args = 0
        filter_events = []
        for e in s["events"]:
            tgt_st, tgt_end = cls.get_tgt_span(
                e["trigger"]["start"] + 1, e["trigger"]["end"] + 1, alignments
            )
            tgt = ""
            if tgt_st > 0:
                for t in range(tgt_st - 1, tgt_end):
                    tgt += " " + segments_tgt[t]
            e["trigger"]["text"] = tgt.strip()
            if tgt_st > 0:
                e["trigger"]["start"] = tgt_st - 1
            else:
                e["trigger"]["start"] = 0
            e["trigger"]["end"] = tgt_end
            if e["trigger"]["text"] == "":
                filter_events.append(e["event_id"])
            filter_event_args = []
            orig_args += len(e["arguments"])
            for a in e["arguments"]:
                a["text"] = cls.update_arg(
                    s["tokens"], a["text"], segments_tgt, alignments
                ).strip()
                if a["text"] == "" or a["mention_id"] in filter_entities:
                    filter_event_args.append(a)
            new_event_args = [x for x in e["arguments"] if x not in filter_event_args]
            saved_args += len(new_event_args)
            e["arguments"] = new_event_args
        return orig_args, saved_args, filter_events

    @classmethod
    def project_label(
        cls, example: Dict, translation: List[str], mapping: List[Tuple]
    ) -> Dict:
        # span projection
        filter_sent = []
        prev_end = 0
        for jj, s in enumerate(example["sentences"]):
            filter_entities = cls.get_filter_ents(
                s["entities"], translation[jj], mapping[jj]
            )
            new_entities = [
                x for x in s["entities"] if not x["mention_id"] in filter_entities
            ]
            ents_saved = 1.0
            if len(s["entities"]) > 0:
                ents_saved = len(new_entities) / len(s["entities"])
            s["entities"] = new_entities

            filter_relation = cls.get_filter_rels(
                s, filter_entities, translation[jj], mapping[jj]
            )
            new_relations = [x for x in s["relations"] if x not in filter_relation]
            rels_saved = 1.0
            if len(s["relations"]) > 0:
                rels_saved = len(new_relations) / len(s["relations"])
            s["relations"] = new_relations

            orig_args, saved_args, filter_events = cls.get_filter_evts(
                s, filter_entities, translation[jj], mapping[jj]
            )
            evt_args_saved = 1.0
            if orig_args > 0:
                evt_args_saved = saved_args / orig_args
            new_events = [x for x in s["events"] if not x["event_id"] in filter_events]
            evts_saved = 1.0
            if len(s["events"]) > 0:
                evts_saved = len(new_events) / len(s["events"])
            s["events"] = new_events

            s["tokens"] = translation[jj]
            s["text"] = TreebankWordDetokenizer().detokenize(s["tokens"])
            s["start"] = prev_end + 1
            s["end"] = s["start"] + len(s["text"])
            prev_end = s["end"]

            if (
                ents_saved == 1
                and rels_saved == 1
                and evts_saved == 1
                and evt_args_saved == 1
            ):
                pass
            else:
                filter_sent.append(s["sent_id"])

        # new_sents = [x for x in example["sentences"] if not x["sent_id"] in filter_sent]
        # if less_filter:
        #     example["sentences"] = new_sents

        return example

    @classmethod
    def get_file(cls, path: str, lang: str, split: str) -> Optional[str]:
        return f"{path}/english.json"

    @classmethod
    def write_example(cls, example: Dict, file_handler):
        file_handler.write(json.dumps(example) + "\n")
