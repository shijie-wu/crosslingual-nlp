# This script receives a bpjson file and tokenizes the
# span strings and segment texts.

import argparse
import json
import os

import tokenizations
from sacremoses import MosesTokenizer

mt = MosesTokenizer(lang="en")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bpjson", type=str, required=True, help="input .bp.json file."
    )
    parser.add_argument(
        "--output", type=str, required=True, help="output .bp.json file."
    )

    args = parser.parse_args()

    return args


def _first_non_empty(tokens):
    for tok in tokens:
        if len(tok) > 0:
            return tok[0]


def find_index(text_tokens, span_tokens, char2tok, char_offset_start, char_offset_end):
    overlapping_tokens = char2tok[char_offset_start:char_offset_end]
    # Ignore mappings to non-tokens (e.g., spaces), which have value `[]`
    # Get the first and last "non empty" alignments
    predicted_token_offset_start = _first_non_empty(overlapping_tokens)
    predicted_token_offset_end = _first_non_empty(overlapping_tokens[::-1])

    match = (predicted_token_offset_start, predicted_token_offset_end)

    assert "".join(span_tokens) in "".join(
        text_tokens[match[0] : match[1] + 1]
    ), "Spans did not match"

    if match[1] - match[0] + 1 != len(span_tokens):
        # Slight offset issue due to character offsets, keep start of span and modify end of span
        match = (match[0], match[0] + len(span_tokens) - 1)

    assert match[1] - match[0] + 1 == len(
        span_tokens
    ), "lens did not match"  # length should match

    return match


def _findall(string, span):
    idx = string.find(span)
    while idx != -1:
        yield idx
        idx = string.find(span, idx + 1)
    return


def align(sentence, span):
    """align. Searches for span in sentence, to create character offsets
    Uses heuristics if the span happens more than once in a sentence.
    Prefers the earliest span that also ends on a word boundary, which
    is defined as a non-alpha numeric character (e.g. a space character)

    returns a tuple (start, end)

    :param sentence: A string sentence
    :param span: The desired text span
    """
    start_idxs = list(_findall(sentence, span))
    idxs = [(i, i + len(span)) for i in start_idxs]

    if len(idxs) == 1:
        return idxs[0]
    elif len(idxs) > 1:
        # heuristic!
        # get the spans that also end on word boundaries
        good_spans = []
        for start, end in idxs:
            end_ok = (end >= len(sentence)) or (not sentence[end].isalpha())
            start_ok = (start - 1 < 0) or (not sentence[start - 1].isalpha())
            if start_ok and end_ok:
                good_spans.append((start, end))

        return good_spans[0]
    else:
        raise Exception("No matches found for this span")


def tokenize_str(text):
    text = f"DUMMYWORD {text} DUMMYWORD"
    text_tok_dir = mt.tokenize(text, escape=False)
    text_tok = []
    for c in text_tok_dir:
        if c not in ["\u202b", "\u202c"]:
            text_tok.append(c)
    return text_tok[1:-1]


def tokenize_data(DATA_FILE, args):
    if not os.path.isfile(DATA_FILE):
        raise FileNotFoundError(f"Could not find {DATA_FILE}")

    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    entries = data["entries"]
    dirty_docs = []
    for entry_id, entry in entries.items():
        entry["segment-text-tok"] = tokenize_str(entry["segment-text"])

        if "annotation-sets" not in entry:
            entry["annotation-sets"] = {
                "abstract-events": {"events": {}, "span-sets": {}}
            }

        annotation_sets = entry["annotation-sets"]

        events = annotation_sets["abstract-events"]

        tok2char, char2tok = tokenizations.get_alignments(
            entry["segment-text-tok"], [char for char in entry["segment-text"]]
        )

        for span_set_id, span_set in events["span-sets"].items():
            spans = span_set["spans"]
            for span in spans:
                span["string-tok"] = tokenize_str(span["string"])
                if "hstring" in span:
                    span["hstring-tok"] = tokenize_str(span["hstring"])

                try:
                    # if this is abstract, we need to extract offsets
                    char_start, char_end = align(entry["segment-text"], span["string"])
                    span["start"] = char_start
                    span["end"] = char_end

                    # now get the token offsets
                    span_indices = find_index(
                        entry["segment-text-tok"],
                        span["string-tok"],
                        char2tok,
                        span["start"],
                        span["end"],
                    )

                    start, end = span_indices
                    span["start-token"] = start
                    span["end-token"] = end
                except Exception:
                    dirty_docs.append(entry["doc-id"])

    dropped = 0
    # original_len = len(data["entries"])
    for doc_id in dirty_docs:
        for entry_id, entry in list(data["entries"].items()):
            if entry["doc-id"] == doc_id:
                dropped += 1
                del data["entries"][entry_id]
    return data


def main():
    args = parse_args()

    # tokenize segments and span strings in input json file
    tok_data = tokenize_data(args.bpjson, args)

    # write tokenization results into a new bpjson file
    with open(args.output, "w", encoding="utf8") as outfile:
        json.dump(tok_data, outfile, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
