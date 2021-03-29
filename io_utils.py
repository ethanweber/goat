import json
from pathlib import Path
import os
from os.path import join as pjoin


def load_from_json(filename: str):
    assert filename.endswith(".json")
    with open(filename, "r") as f:
        return json.load(f)


def write_to_json(filename: str, content: dict):
    assert filename.endswith(".json")
    with open(filename, "w") as f:
        json.dump(content, f)


def get_project_root() -> Path:
    return Path(__file__).parent.parent.absolute()


def get_absolute_path(path):
    """
    Returns the full, absolute path.
    Relative paths are assumed to start at the repo directory.
    """
    absolute_path = path
    if absolute_path[0] != "/":
        absolute_path = os.path.join(
            get_project_root(), absolute_path
        )
    return absolute_path


def make_dir(filename_or_folder):
    """Make the directory for either the filename or folder.
    Note that filename_or_folder currently needs to end in / for it to be recognized as a folder.
    """
    folder = os.path.dirname(filename_or_folder)
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except Exception as e:
            print(f"Couldn't create folder: {folder}. Maybe due to a parallel process?")
            print(e)
