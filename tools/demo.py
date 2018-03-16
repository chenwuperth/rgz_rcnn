#    ICRAR - International Centre for Radio Astronomy Research
#    (c) UWA - The University of Western Australia, 2017
#    Copyright by UWA (in the framework of the ICRAR)
#    All rights reserved
#
#    This library is free software; you can redistribute it and/or
#    modify it under the MIT license
#
#    Created on 15 March 2018 by chen.wu@icrar.org

import os, sys
import os.path as osp
import argparse
from itertools import cycle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from astropy.io import fits
import tensorflow as tf

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect, remove_embedded
from fast_rcnn.nms_wrapper import nms
from networks.factory import get_network
from utils.timer import Timer
from download_data import get_rgz_root
from fuse_radio_ir import fuse

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Run a demo detector')
    parser.add_argument('--device', dest='device', help='device to use',
                        default='cpu', type=str)
    parser.add_argument('--device_id', dest='device_id', help='device id to use for GPUs',
                        default=0, type=int)
    parser.add_argument('--radio', dest='radio_fits',
                        help='full path of the radio fits file (compulsory)',
                        default=None, type=str)
    parser.add_argument('--ir', dest='ir_png',
                        help='full path of the infrared png file (compulsory)',
                        default=None, type=str)

    args = parser.parse_args()
    if (args.radio_fits is None or args.ir_png is None):
        parser.print_help()
        sys.exit(1)

    if (not osp.exists(args.radio_fits)):
        print('Radio fits %s not found' % args.radio_fits)
        sys.exit(1)

    if (not osp.exists(args.ir_png)):
        print('Infrared png %s not found' % args.ir_png)
        sys.exit(1)

    if args.device.lower() == 'gpu':
        cfg.USE_GPU_NMS = True
        cfg.GPU_ID = args.device_id
        device_name = '/{}:{:d}'.format(args.device, args.device_id)
        print(device_name)
    else:
        cfg.USE_GPU_NMS = False

    return args

def hard_code_cfg():
    """
    This could be parameters potentially. Hardcoded for now
    """
    cfg.TEST.HAS_RPN = True
    cfg.TEST.RPN_MIN_SIZE = 4
    cfg.TEST.RPN_POST_NMS_TOP_N = 5
    cfg.TEST.NMS = 0.3
    cfg.TEST.RPN_NMS_THRESH = 0.5

def fuse_radio_ir(radio_fn, ir_fn, out_dir='/tmp'):
    """
    return the full path to the generated fused file
    """
    return fuse(radio_fn, ir_fn, out_dir)

if __name__ == '__main__':
    args = parse_args()
    fuse_radio_ir(args.radio_fits, args.ir_png)
    net = get_network('rgz_test')
    hard_code_cfg()
    model_weight = osp.join(get_rgz_root(), 'data/pretrained_model/rgz/D4/VGGnet_fast_rcnn-80000')
    if (not osp.exists(model_weight + '.index')):
        print("Fail to load rgz model, have you run python download_data.py already?")
        sys.exit(1)

    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    print('Loading model weights from {:s}').format(model_weight)
    saver.restore(sess, model_weight)
