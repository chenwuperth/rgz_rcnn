# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
#
# Modified by Chen Wu (chen.wu@icrar.org)
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import datasets.rgz
import numpy as np

# RGZ dataset
for year in ['2017']:
    for split in ['trainD1', 'testD1',
                  'trainD3', 'testD3',
                  'trainD4', 'testD4',
                  'trainD5', 'testD5',
				  'allD1']:
        name = 'rgz_{}_{}'.format(year, split)
        print('Loading dataset %s' % name)
        __sets[name] = (lambda split=split, year=year:
                datasets.rgz(split, year))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
