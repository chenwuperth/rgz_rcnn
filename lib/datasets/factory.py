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
    for split in ['train', 'test', 'trainsecond',
                  'trainthird', 'testthird', 'trainthirdsmall',
                  'trainfourth', 'testfourth', 'testthirdsmall',
                  'trainfifth', 'testfifth', 'trainsixth', 'testsixth',
                  'train07', 'test07', 'train08', 'test08',
                  'train09', 'test09', 'train10', 'test10',
                  'train11', 'test11', 'train12', 'test12',
                  'train13', 'test13', 'train14', 'test14',
                  'train15', 'test15', 'train16', 'test16',
                  'train17', 'test17', 'train18', 'test18',
                  'train19', 'test19', 'train20', 'test20',
                  'train21', 'test21', 'train22', 'test22']:
        name = 'rgz_{}_{}'.format(year, split)
        print name
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
