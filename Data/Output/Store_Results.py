import os
import pickle

from pathlib import Path
from collections import defaultdict


OUTPUT_DIR = Path("Data/Output")
CHECKPOINT_FILE = OUTPUT_DIR / "pipeline_state.pkl"


def merge_linked_annotations(
    all_linked_annotations,
    new_linked_annotations
):
    for reference_name, annotations in new_linked_annotations.items():
        all_linked_annotations[reference_name].extend(annotations)


def load_pipeline_state():
    if not CHECKPOINT_FILE.exists():
        return {
            "linked_annotations": defaultdict(list),
            "completed_archives": set(),
            "seen_pmids": set(),
            "seen_pmcs": set(),
        }

    with open(CHECKPOINT_FILE, "rb") as fp:
        state = pickle.load(fp)

    state["linked_annotations"] = defaultdict(
        list,
        state["linked_annotations"]
    )

    return state


def save_pipeline_state(state):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    temp_file = CHECKPOINT_FILE.with_suffix(".tmp")

    with open(temp_file, "wb") as fp:
        pickle.dump(state, fp)

    os.replace(temp_file, CHECKPOINT_FILE)