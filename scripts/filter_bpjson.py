import argparse
import json
import traceback
from copy import deepcopy
from pathlib import Path

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input .bp.json file.")
    parser.add_argument(
        "--outputdir", type=str, required=True, help="output directory."
    )
    args = parser.parse_args()
    return args


def string_check(value: dict) -> bool:
    # check if does not exist
    if "string" not in value:
        return False

    # check for empty string
    if value["string"] == "":
        return False

    # check tok for being too long (more than 40 tokens)
    if len(value["string"].split(" ")) > 40:
        return False

    return True


def hstring_check(value: dict) -> bool:
    # check if does not exist
    if "hstring" not in value:
        return False

    # check for empty string
    if value["hstring"] == "":
        return False

    # check tok for being too long (more than 18 tokens)
    if len(value["hstring"].split(" ")) > 18:
        return False

    return True


def spansets_check(value: dict):
    valid_spans = []
    # check if empty
    if not value:
        return []

    # for each spanset...
    for spanset in value.keys():
        # for each of a spansets' span...
        spans_valid_items = []
        for span in value[spanset]["spans"]:
            # see if they are valid, if one isn't, return False
            check = hstring_check(span)
            check &= string_check(span)

            if check:
                spans_valid_items.append(span)

        if spans_valid_items != []:
            value[spanset]["spans"] = spans_valid_items
            valid_spans.append(spanset)

    return valid_spans


def process_file(file_p: Path, dir_s: Path):
    with open(str(file_p)) as json_file:
        data = None

        # if file is empty/not proper json
        try:
            data = json.load(json_file)
        except json.JSONDecodeError:
            traceback.print_exc()
            json_file.close()
            return

        # copy data to valid/filtered
        valid = deepcopy(data)
        valid["entries"] = {}

        # for every entry
        for entry in tqdm(data["entries"], desc="entries"):
            entry_data = data["entries"][entry]

            valid_spans = spansets_check(
                entry_data["annotation-sets"]["abstract-events"]["span-sets"]
            )
            spansets = entry_data["annotation-sets"]["abstract-events"]["span-sets"]

            valid_spansets = {}
            for spanset_k, spanset in spansets.items():
                if spanset_k in valid_spans:
                    valid_spansets[spanset_k] = spanset
            entry_data["annotation-sets"]["abstract-events"][
                "span-sets"
            ] = valid_spansets
            events = entry_data["annotation-sets"]["abstract-events"]["events"]

            valid_events = {}
            for event_k, event in events.items():
                e = {}
                e_k = {}

                if event["anchors"] in valid_spans:
                    e_agents = []
                    for a in event["agents"]:
                        if a in valid_spans:
                            e_agents.append(a)
                    e["agents"] = e_agents
                    e["anchors"] = event["anchors"]
                    e["eventid"] = event["eventid"]
                    e_patients = []
                    for p in event["patients"]:
                        if p in valid_spans:
                            e_patients.append(p)
                    if "event-type" in event:
                        e["event-type"] = event["event-type"]
                    if "helpful-harmful" in event:
                        e["helpful-harmful"] = event["helpful-harmful"]
                    if "material-verbal" in event:
                        e["material-verbal"] = event["material-verbal"]
                    e["patients"] = e_patients

                if e:
                    e_k[event_k] = e
                    valid_events.update(e_k)

            entry_data["annotation-sets"]["abstract-events"]["events"] = valid_events

            valid["entries"][entry] = entry_data
        print("number of valid entries:", len(valid["entries"]))

        # save file
        with open(str(dir_s / (file_p.stem + ".valid.bp.json")), "w") as valid_json:
            json.dump(valid, valid_json, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    args = parse_args()

    file_to_open = Path(args.input)
    directory_to_save = Path(args.outputdir)
    process_file(file_to_open, directory_to_save)
