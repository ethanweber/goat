import json
from pathlib import Path
import os
from os.path import join as pjoin
import pickle
import yaml


def load_from_json(filename: str):
    assert filename.endswith(".json")
    with open(filename, "r") as f:
        return json.load(f)


def write_to_json(filename: str, content: dict):
    assert filename.endswith(".json")
    with open(filename, "w") as f:
        json.dump(content, f)


def load_from_pkl(filename: str):
    assert filename.endswith(".pkl")
    with open(filename, "rb") as f:
        return pickle.load(f)


def write_to_pkl(filename: str, content):
    assert filename.endswith(".pkl")
    with open(filename, "wb") as f:
        pickle.dump(content, f)


def load_from_txt(filename: str):
    assert filename.endswith(".txt")
    with open(filename) as f:
        lines = f.readlines()
        return lines


def write_to_txt(filename: str, content):
    assert filename.endswith(".txt")
    with open(filename, "w") as f:
        f.write(content)


def get_git_root(path, dirs=(".git",), default=None):
    """https://stackoverflow.com/questions/22081209/find-the-root-of-the-git-repository-where-the-file-lives
    """
    import os
    prev, test = None, os.path.abspath(path)
    while prev != test:
        if any(os.path.isdir(os.path.join(test, d)) for d in dirs):
            return test
        prev, test = test, os.path.abspath(os.path.join(test, os.pardir))
    return default


def get_absolute_path(path, proj_root_func=get_git_root):
    """
    Returns the full, absolute path.
    Relative paths are assumed to start at the repo directory.
    """
    if path == "":
        return ""
    absolute_path = path
    if absolute_path[0] != "/":
        absolute_path = os.path.join(
            proj_root_func(path), absolute_path
        )
    return absolute_path


def make_dir(filename_or_folder):
    """Make the directory for either the filename or folder.
    Note that filename_or_folder currently needs to end in / for it to be recognized as a folder.
    """
    if filename_or_folder[-1] != "/" and filename_or_folder.find(".") < 0:
        folder = os.path.dirname(filename_or_folder + "/")
    else:
        folder = os.path.dirname(filename_or_folder)
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except Exception as e:
            print(f"Couldn't create folder: {folder}. Maybe due to a parallel process?")
            print(e)
    return filename_or_folder
