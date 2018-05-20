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
import _init_paths

import os
import cPickle
import os.path as osp
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# classes that characterise both components and peaks
CLASSES =  ('__background__', # always index 0
                            '1_1', '1_2', '1_3', '2_2', '2_3', '3_3')

COMP_CLASSES = ['__background__', '1_', '2_', '3_'] # component class only

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

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

def filter_detections(detpath, outpath, threshold=0.8):
    """
    Given a subject (image),
        remove all sources whose Probability is less than threshold
        if none sources are left, retain the one with the highest Probability
    """
    all_dets = defaultdict(list) # key: image_name, value: a list of dets
    out_list = defaultdict(list) # key: class_name, value: a list of det strings
    for classname in CLASSES[1:]:
        detfile = detpath.format(classname)
        with open(detfile, 'r') as f:
            lines = f.readlines()
        splitlines = [x.strip().split(' ') for x in lines]
        for det in splitlines:
            det.append(classname)
            all_dets[det[0]].append(det)

    for _, v in all_dets.items():
        v.sort(key=lambda x: -float(x[1])) # sort on descending order
        if (float(v[0][1]) < threshold):
            out_list[v[0][-1]].append(' '.join(v[0][0:-1]))
        else:
            for vi in v:
                if (float(vi[1]) >= threshold):
                    #print(vi[1])
                    out_list[vi[-1]].append(' '.join(vi[0:-1]))

    for classname, v in out_list.items():
        outfile = outpath.format(classname)
        with open(outfile, 'w') as fout:
            fout.write(os.linesep.join(v))

def compute_map_from_subset(imagesetfile, anno_file, detpath, catalog_csv, subset_file, ovthresh=0.5):
    """
    Get mAP score from a subset of the FirstIDs

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
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    with open(subset_file, 'r') as fi:
        subsets = fi.readlines()
    subsets = set([x.strip() for x in subsets])
    #print(subsets)

    #for classname in CLASSES[1:]:
    for classname in COMP_CLASSES[1:]:
        class_recs = dict() # class-specific ground truth
        npos = 0
        for imagename in imagenames:
            if (not imagename.split('_')[0] in subsets):
                continue
            R = [obj for obj in recs[imagename] if obj['name'].startswith(classname)]
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imagename] = {'bbox': bbox,
                                     'difficult': difficult,
                                     'det': det}
        # read dets
        # detfile = detpath.format(classname)
        # with open(detfile, 'r') as f:
        #     all_lines = f.readlines()

        # filter based on subsets
        # lines = all_lines
        def filter_lines(all_lines_p):
            lines = []
            for x in all_lines_p:
                sp = x.strip().split(' ')
                #print(sp)
                first_id = sp[0].split('_')[0]
                score = float(sp[1])
                if (first_id in subsets and score > 0.0):
                    #print("got it", first_id)
                    lines.append(x)
            return lines

        #lines = filter_lines(all_lines)

        # merge into a big list e.g. 2C_2P and 2C_3P will join the 2C list
        lines = []
        for c_p_cls in CLASSES:
            if (c_p_cls.startswith(classname)):
                detfile = detpath.format(c_p_cls)
                with open(detfile, 'r') as f:
                    lines_c = f.readlines()
                lines += filter_lines(lines_c)

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
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for d in range(nd):
                R = class_recs[image_ids[d]]
                bb = BB[d, :].astype(float)
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)

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
                        if not R['det'][jmax]:
                            tp[d] = 1.
                            R['det'][jmax] = 1
                        else:
                            fp[d] = 1.
                else:
                    fp[d] = 1.

            # compute precision recall
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = voc_ap(rec, prec)
            print(classname, ap)
    else:
         rec = -1
         prec = -1
         ap = -1

#TODO refactor the following two functions due to overlapping
def get_component_map(imagesetfile, anno_file, detpath, catalog_csv, ovthresh=0.5):
    """
    Get component-only mAP score

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
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    for classname in COMP_CLASSES[1:]:
        class_recs = dict() # class-specific ground truth
        npos = 0
        for imagename in imagenames:
            R = [obj for obj in recs[imagename] if obj['name'].startswith(classname)]
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imagename] = {'bbox': bbox,
                                     'difficult': difficult,
                                     'det': det}

        # merge into a big list e.g. 2C_2P and 2C_3P will join the 2C list
        lines = []
        for c_p_cls in CLASSES:
            if (c_p_cls.startswith(classname)):
                detfile = detpath.format(c_p_cls)
                with open(detfile, 'r') as f:
                    lines_c = f.readlines()
                lines += lines_c

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
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for d in range(nd):
                R = class_recs[image_ids[d]]
                bb = BB[d, :].astype(float)
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)

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
                        if not R['det'][jmax]:
                            tp[d] = 1.
                            R['det'][jmax] = 1
                        else:
                            fp[d] = 1.
                else:
                    fp[d] = 1.

            # compute precision recall
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = voc_ap(rec, prec)
            print(classname, ap)
    else:
         rec = -1
         prec = -1
         ap = -1

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

def plot_prob_cl_box(prob_cl_mapping_list, plot_outliers=False):
    ks = prob_cl_mapping_list.keys()
    ks.sort()
    for i, classname in enumerate(ks):
        v = prob_cl_mapping_list[classname]
        data = [[], [], [], []]
        labels = ['0.6~0.7', '0.7~0.8', '0.8~0.9', '0.9~1.0']
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
        if (plot_outliers):
            symb = '+'
        else:
            symb = ''
        plt.boxplot(data, labels=labels, sym=symb)
        plt.xlabel('Consensus level')
        plt.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5, axis='y')
        if (i % 2 == 0):
            #plt.ylabel('Classification probability')
            plt.ylabel('Probability')
        # if (not plot_outliers):
        #     plt.ylim([0.6, 1.0])
        ax.set_title('%s' % classname.replace('_', 'C_') + 'P')
    #plt.suptitle('Probability vs. Consensus level')
    plt.tight_layout(h_pad=0.0)
    plt.show()

if __name__ == '__main__':
    rgz_cnn_data = '/Users/Chen/proj/rgz_rcnn/data/RGZdevkit2017' #TODO passed in as an argument
    model_v = 3
    imagesetfile = osp.join(rgz_cnn_data,
                        'RGZ2017/ImageSets/Main/testD%d.txt' % model_v)
    anno_file = osp.join(rgz_cnn_data,
                        'annotations_cache/annots_D%d.pkl' % model_v)
    catalog_csv = osp.join(rgz_cnn_data,
                        'RGZ2017/ImageSets/Main/full_catalogue.csv')
    #outpath = osp.join(rgz_cnn_data,
    #                    'results/RGZ2017/Filtered/comp4_det_testD4_{0}.txt')
    #detpath = outpath
    detpath = osp.join(rgz_cnn_data,
                         'results/RGZ2017/Main/comp4_det_testD%d_{0}.txt' % model_v)
    subsetf = osp.join(rgz_cnn_data,
                         'RGZ2017/ImageSets/Main/multisource.txt')

    #filter_detections(detpath, outpath)
    #ret = get_prob_cl_mapping_list(imagesetfile, anno_file, detpath, catalog_csv)
    #plot_prob_cl_box(ret)
    #get_component_map(imagesetfile, anno_file, detpath, catalog_csv)
    compute_map_from_subset(imagesetfile, anno_file, detpath, catalog_csv,
                            subset_file=subsetf, ovthresh=0.5)

    # for k, v in ret.items():
    #     print(k, len(v[0]), len(v[1]))
    #     plot_prob_cl_corr(k, v[0], v[1])
