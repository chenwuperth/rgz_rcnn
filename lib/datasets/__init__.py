# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
#
# Modified by Chen Wu
# --------------------------------------------------------

from .imdb import imdb
from .pascal_voc import pascal_voc
from .rgz import rgz
from . import factory

import os.path as osp
ROOT_DIR = osp.join(osp.dirname(__file__), '..', '..')
