#    ICRAR - International Centre for Radio Astronomy Research
#    (c) UWA - The University of Western Australia, 2017
#    Copyright by UWA (in the framework of the ICRAR)
#    All rights reserved
#
#    This library is free software; you can redistribute it and/or
#    modify it under the MIT license
#
#    Created on 17 June 2017 by chen.wu@icrar.org

import os
import uuid
import cPickle
import numpy as np

from datasets.imdb import imdb
from datasets.pascal_voc import pascal_voc
from fast_rcnn.config import cfg
from voc_eval import voc_eval

"""
Prepare data sets for the RGZ project
"""

class rgz(pascal_voc):
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'rgz_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'RGZ' + self._year)
        self._classes = ('__background__',  # always index 0
                         '1_1', '1_2', '1_3', '2_2', '2_3', '3_3')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        if image_set in ['train21', 'test21', 'train22', 'test22']:
            self._image_ext = '.png'
        else:
            self._image_ext = '_radio.png'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        #self._roidb_handler = self.selective_search_roidb
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # RGZ specific config options
        self.config = {'cleanup': False,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 2}

        assert os.path.exists(self._devkit_path), \
            'RGZdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'PNGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _get_default_path(self):
        """
        Return the default path where RGZ is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'RGZdevkit' + self._year)

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/RGZ2017/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + \
            self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            'RGZ' + self._year,
            'Main',
            filename)
        return path

    def _do_python_eval(self, output_dir='output'):
        annopath = os.path.join(
            self._devkit_path,
            'RGZ' + self._year,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'RGZ' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
