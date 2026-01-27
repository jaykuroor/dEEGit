"""
dEEGit: EEG-Based Digit Classification using Deep Learning

A package for training and evaluating EEGNet models on EEG data
recorded during digit stimulus presentation.
"""

from . import config
from . import parse
from . import dataset
from . import model
from . import train
from . import eval

__version__ = "0.1.0"
__all__ = ['config', 'parse', 'dataset', 'model', 'train', 'eval']
