#    ICRAR - International Centre for Radio Astronomy Research
#    (c) UWA - The University of Western Australia, 2017
#    Copyright by UWA (in the framework of the ICRAR)
#    All rights reserved
#
#    This library is free software; you can redistribute it and/or
#    modify it under the MIT license
#
#    Created on 26 Arpil 2018 by chen.wu@icrar.org

"""
This script compares the class probabilities produced by Claran and the RGZ
Consensus Level (CL) for each source. Note that in the current RGZ truth,
all sources in the same subject share the same CL.
"""

import os
import cPickle
import os.path as osp

import numpy as np
import matplotlib.pyplot as plt

CLASSES =  ('__background__', # always index 0
                            '1_1', '1_2', '1_3', '2_2', '2_3', '3_3')

def load_annotations(cachefile):
    if (not osp.exists(cachefile)):
        raise Exception("Annotation cache file not found: %s" % cachefile)
    with open(cachefile, 'r') as f:
        recs = cPickle.load(f)
    return recs

def load_cl(catalog_csv, suffix='_infraredctmask'):
    """
    catalog_csv:    e.g
                    data/RGZdevkit2017/RGZ2017/ImageSets/Main/full_catalogue.csv
    """
    with open(catalog_csv, 'r') as f:
        lines = f.readlines()[1:] # skip headers

    catalogs = [x.strip() for x in lines]
    kv = dict()
    for catline in catalogs:
        cat = catline.split(',')
        kv[cat[1] + suffix] = float(cat[-2])
    return kv

def get_prob_cl_mapping_list(imagesetfile, anno_file, detpath, catalog_csv, ovthresh=0.5):
    """
    imagesetfile:   Text file containing the list of images, one image per line
                    e.g. data/RGZdevkit2017/RGZ2017/ImageSets/Main/testD4.txt

    anno_file:      annotation pickle file (i.e. the groud-truth)
                    e.g. data/RGZdevkit2017/annotations_cache/annots.pkl

    detpath:        Path to detections
                    e.g. data/RGZdevkit2017/results/RGZ2017/Main/comp4_det_testD4_{0}.txt

    catalog_csv:    e.g
                    data/RGZdevkit2017/RGZ2017/ImageSets/Main/full_catalogue.csv
    """
    recs = load_annotations(anno_file) # ground-truth for all classes
    cl_dict = load_cl(catalog_csv)
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    ret = dict()
    for classname in CLASSES[1:]:
        class_recs = dict() # class-specific ground truth
        ret_list = [[], []]
        for imagename in imagenames:
            R = [obj for obj in recs[imagename] if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            class_recs[imagename] = {'bbox': bbox,
                                     'difficult': difficult,
                                     'det': det,
                                     'cl': cl_dict[imagename]}
        # read dets
        detfile = detpath.format(classname)
        with open(detfile, 'r') as f:
            lines = f.readlines()
        if any(lines) == 1:
            splitlines = [x.strip().split(' ') for x in lines]
            image_ids = [x[0] for x in splitlines]
            confidence = np.array([float(x[1]) for x in splitlines])
            BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            nd = len(image_ids)
            for d in range(nd):
                R = class_recs[image_ids[d]] # gt
                bb = BB[d, :].astype(float) # detected
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float) # gt

                if BBGT.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    # union
                    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                           (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                           (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if not R['difficult'][jmax]:
                        # if this source is taken, then the detection is FP!
                        if not R['det'][jmax]:
                            ret_list[0].append(-sorted_scores[d])
                            ret_list[1].append(R['cl'])
        ret[classname] = ret_list

    return ret

def plot_prob_cl_corr(classname, prob_list, cl_list):
    plt.scatter(cl_list, prob_list)
    plt.show()

def plot_prob_cl_box(prob_cl_mapping_list):
    ks = prob_cl_mapping_list.keys()
    ks.sort()
    for i, classname in enumerate(ks):
        v = prob_cl_mapping_list[classname]
        data = [[], [], [], []]
        labels = ['[0.6, 0.7)', '[0.7, 0.8)', '[0.8, 0.9)', '[0.9, 1.0]']
        prob_list = v[0]
        cl_list = v[1]
        for cl, prob in zip(cl_list, prob_list):
            if 0.6 <= cl < 0.7:
                ind = 0
            elif 0.7 <= cl < 0.8:
                ind = 1
            elif 0.8 <= cl < 0.9:
                ind = 2
            elif 0.9 <= cl <= 1.0:
                ind = 3
            else:
                raise Exception("invalid CL: %.3f" % cl)
            data[ind].append(prob)
        ax = plt.subplot(3, 2, i + 1)
        plt.boxplot(data, labels=labels, sym='+')
        plt.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5, axis='y')
        plt.ylabel('Probability')
        ax.set_title('Prob vs. CL for %s' % classname)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    rgz_cnn_data = '/Users/Chen/proj/rgz_rcnn/data/RGZdevkit2017' #TODO passed in as an argument
    imagesetfile = osp.join(rgz_cnn_data,
                        'RGZ2017/ImageSets/Main/testD4.txt')
    anno_file = osp.join(rgz_cnn_data,
                        'annotations_cache/annots.pkl')
    detpath = osp.join(rgz_cnn_data,
                        'results/RGZ2017/Main/comp4_det_testD4_{0}.txt')
    catalog_csv = osp.join(rgz_cnn_data,
                        'RGZ2017/ImageSets/Main/full_catalogue.csv')
    ret = get_prob_cl_mapping_list(imagesetfile, anno_file, detpath, catalog_csv)
    plot_prob_cl_box(ret)
    # for k, v in ret.items():
    #     print(k, len(v[0]), len(v[1]))
    #     plot_prob_cl_corr(k, v[0], v[1])
