# --------------------------------------------------------
# SubCNN_TF
# Copyright (c) 2016 CVGL Stanford
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
#
# Modified by Chen Wu (chen.wu@icrar.org)
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import networks.VGGnet_train
import networks.VGGnet_test
import pdb
import tensorflow as tf

#__sets['VGGnet_train'] = networks.VGGnet_train()

#__sets['VGGnet_test'] = networks.VGGnet_test()

def get_network(name):
    """Get a network by name."""
    nwnm = name.split('_')[1]

    if nwnm.find('train') > -1:
        #return networks.VGGnet_test()
        return networks.VGGnet_train(low_level_trainable=False,
                                     anchor_scales=[1, 2, 4, 8, 16, 32],
                                     anchor_ratios=[1], transform_img=True)
    elif nwnm.find('test') > -1:
        #return networks.VGGnet_train()
        return networks.VGGnet_test(low_level_trainable=False,
                                     anchor_scales=[1, 2, 4, 8, 16, 32],
                                     anchor_ratios=[1], transform_img=True)
    else:
        raise KeyError('Unknown dataset: {}'.format(name))

def list_networks():
    """List all registered imdbs."""
    return __sets.keys()
