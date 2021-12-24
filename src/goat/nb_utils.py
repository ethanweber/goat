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
import pickle

# image stuff
import imageio
import mediapy as media
import cv2

# ipython stuff
from IPython import get_ipython
def setup_ipynb():
    """
    https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    https://stackoverflow.com/questions/35595766/matplotlib-line-magic-causes-syntaxerror-in-python-script
    This gets reference to the InteractiveShell instance
    """
    try:
        from IPython import get_ipython
        get_ipython().run_line_magic('load_ext', 'autoreload')
        get_ipython().run_line_magic('autoreload', '2')
        get_ipython().run_line_magic('matplotlib', 'inline')
        return True
    except:
        return False

setup_ipynb()