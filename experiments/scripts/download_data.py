#    ICRAR - International Centre for Radio Astronomy Research
#    (c) UWA - The University of Western Australia, 2017
#    Copyright by UWA (in the framework of the ICRAR)
#    All rights reserved
#
#    This library is free software; you can redistribute it and/or
#    modify it under the MIT license
#
#    Created on 11 March 2018 by chen.wu@icrar.org

"""
Need to create these directories

This script is Python 2 compatible ONLY

data/RGZdevkit2017/RGZ2017/Annotations
data/RGZdevkit2017/RGZ2017/PNGImages
data/pretrained_model

data/RGZdevkit2017/results/RGZ2017/Main
output/faster_rcnn_end2end
"""

import os
import os.path as osp
import commands
import time

rgz_data_url = 'http://ict.icrar.org/store/staff/cwu/rgz_data'
rgz_dn_dict = {'d1_img': 'D1_images.tgz',
               'd1_model':  'D1_model.tar',
               'd3_img':  'D3_images.tgz',
               'd3_model':  'D3_model.tar',
               'd4_img':  'D4_images.tgz',
               'd4_model':  'D4_model.tar',
               'vgg_weights':  'VGG_imagenet.npy',
               'anno':  'annotation.tar'}

def get_full_url(fkey):
    if (not fkey in rgz_dn_dict):
        raise Exception('key %s unknown' % fkey)
    return '%s/%s' % (rgz_data_url, rgz_dn_dict[fkey])

def check_req():
    cmd = 'wget --help'
    status, msg = commands.getstatusoutput(cmd)
    if (status != 0):
        raise Exception('wget is not installed properly')

def download_file(url, tgt_dir):
    if (not osp.exists(tgt_dir)):
        raise Exception("tgt_dir %s not found" % tgt_dir)
    fn = url.split('/')[-1] # hack hack hack
    full_fn = osp.join(tgt_dir, fn)
    if (osp.exists(full_fn)):
        print("%s exists already, skip downloading" % full_fn)
        return
    cmd = 'wget -O %s %s' % (full_fn, url)
    print("Downloading %s from %s to %s" % (fn, url, tgt_dir))
    stt = time.time()
    status, msg = commands.getstatusoutput(cmd)
    if (status != 0):
        raise Exception("Download %s failed: %s" % (url, msg))
    print("Downloading took %.3f seconds" % (time.time() - stt))
    return osp.join(full_fn, fn)

def extract_file(tar_file, tgt_dir):
    if (not osp.exists(tar_file)):
        raise Exception("file not found for extraction: %s" % tar_file)
    if (not tgt_dir.endswith('/')):
        tgt_dir += '/'
    cmd = 'tar -xf %s -C %s' % (tar_file, tgt_dir)
    print("Extracting from %s to %s" % (tar_file, tgt_dir))
    stt = time.time()
    status, msg = commands.getstatusoutput(cmd)
    if (status != 0):
        raise Exception("fail to extract %s: %s" % (tar_file, msg))
    print("Extraction took %.3f seconds" % (time.time() - stt))

def get_rgz_root():
    # current file directory
    cfd = osp.dirname(osp.abspath(__file__))
    rgz_root = osp.abspath(osp.join(cfd, '..', '..'))
    print("RGZ root: '%s'" % rgz_root)
    return rgz_root

def setup_something(rgz_root, rel_tgt_dir, what, extract=True):
    some_path = osp.join(rgz_root, rel_tgt_dir)
    if (not osp.exists(some_path)):
        os.makedirs(some_path)
    some_file_path = download_file(get_full_url(what), some_path)
    if (extract):
        extract_file(some_file_path, some_path)

def setup_annotations(rgz_root):
    setup_something(rgz_root, 'data/RGZdevkit2017/RGZ2017/Annotations', 'anno')

def setup_png_images(rgz_root):
    png_rel = 'data/RGZdevkit2017/RGZ2017/PNGImages'
    for img_nm in ['d1_img', 'd3_img', 'd4_img']:
        setup_something(rgz_root, png_rel, img_nm)

def setup_models(rgz_root):
    model_rel = 'data/pretrained_model/rgz'
    for model_nm in ['d1_model', 'd3_model', 'd4_model']:
        setup_something(rgz_root, model_rel, model_nm)

def setup_vgg_weights(rgz_root):
    setup_something(rgz_root, 'data/pretrained_model/imagenet', 'vgg_weights',
                    extract=False)

def create_empty_dirs(rgz_root):
    rps = ['data/RGZdevkit2017/results/RGZ2017/Main',
           'output/faster_rcnn_end2end']
    for rp in rps:
        pa = osp.join(rgz_root, rp)
        if (not osp.exists(pa)):
            os.makedirs(pa)

if __name__ == '__main__':
    check_req()
    rr = get_rgz_root()
    setup_annotations(rr)
    setup_png_images(rr)
    setup_models(rr)
    setup_vgg_weights(rr)
    create_empty_dirs()
