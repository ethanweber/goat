"""Specify the imports that are useful for notebook experiments.
"""
import sys
import os
import copy
import random
import cv2
import torch
import pprint
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import glob
from os.path import join as pjoin


from .io_utils import *
from . import debug_utils
from . import view_utils
from .coco_dataset import COCODataset


# https://stackoverflow.com/questions/35595766/matplotlib-line-magic-causes-syntaxerror-in-python-script
# This gets reference to the InteractiveShell instance
from IPython import get_ipython
try:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
    get_ipython().run_line_magic('matplotlib', 'inline')
except:
    pass
