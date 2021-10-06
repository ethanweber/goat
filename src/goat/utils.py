import argparse
import copy
import datetime
import os
import random
import socket
import string
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import GPUtil


def get_gpus(maxLoad=0.5, maxMemory=0.5):
    """Returns the available GPUs."""
    deviceIDs = GPUtil.getAvailable(
        order='first',
        limit=8,
        maxLoad=maxLoad,
        maxMemory=maxMemory,
        includeNan=False,
        excludeID=[],
        excludeUUID=[])
    return deviceIDs


def get_chunks(lst, n):
    """Returns list of n elements, constaining a sublist."""
    size = len(lst) // n
    chunks = []
    for i in range(0, len(lst), size):
        chunks.append(lst[i:i + size])
    return chunks


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return x


def to_torch(x):
    return torch.FloatTensor(x)


def dict_to_torch_device(stuff, torch_device):
    """Set everything in the dict (2 layers deep) to a specified torch device.
    """
    if type(stuff) is dict:
        for k, v in stuff.items():
            stuff[k] = dict_to_torch_device(v, torch_device)
    if type(stuff) is torch.Tensor:
        return stuff.to(torch_device)
    else:
        return stuff


def dict_to_cpu(stuff):
    if type(stuff) is dict:
        for k, v in stuff.items():
            stuff[k] = dict_to_cpu(v)
    if type(stuff) is torch.Tensor:
        return stuff.detach().cpu()
    else:
        return stuff


def dict_to_torch(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_torch(v)
        elif isinstance(v, (np.ndarray, np.generic)):
            d[k] = torch.from_numpy(v)
    return d


def dict_to_numpy(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_numpy(v)
        elif isinstance(v, torch.Tensor):
            d[k] = to_numpy(v)
    return d


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_random_string(length=8):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def gettimedatestring():
    return datetime.datetime.now().strftime("%m-%d-%H-%M-%S")


def select_gpus(gpus_arg):
    # so that default gpu is one of the selected, instead of 0
    if len(gpus_arg) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus_arg
        gpus = list(range(len(gpus_arg.split(','))))
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        gpus = []
    print('CUDA_VISIBLE_DEVICES={}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    return gpus


def str2bool(v):
    assert type(v) is str
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean (yes, true, t, y or 1, lower or upper case) string expected.')


def listdir(folder, prepend_folder=False, extension=None, type=None):
    assert type in [None, 'file', 'folder'], "Type must be None, 'file' or 'folder'"
    files = [k for k in os.listdir(folder) if (True if extension is None else k.endswith(extension))]
    if type == 'folder':
        files = [k for k in files if os.path.isdir(folder + '/' + k)]
    elif type == 'file':
        files = [k for k in files if not os.path.isdir(folder + '/' + k)]
    if prepend_folder:
        files = [folder + '/' + f for f in files]
    return files


def get_hostname():
    return socket.gethostname()
