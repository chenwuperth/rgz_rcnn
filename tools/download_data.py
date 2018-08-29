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
data/rgzdemo
data/pretrained_model

data/RGZdevkit2017/results/RGZ2017/Main
output/faster_rcnn_end2end
"""

import os
import os.path as osp
import subprocess
import time

rgz_data_url = 'http://ict.icrar.org/store/staff/cwu/rgz_data'
rgz_dn_dict = {'d1_img': 'D1_images.tgz',
               'd1_model':  'D1_model.tar',
               'd3_img':  'D3_images.tgz',
               'd3_model':  'D3_model.tar',
               'd4_img':  'D4_images.tgz',
               'd4_model':  'D4_model.tar',
               'vgg_weights':  'VGG_imagenet.npy',
               'anno':  'annotation.tar',
               'demo_img':  'DEMO_images.tgz'}

def get_full_url(fkey):
    if (not fkey in rgz_dn_dict):
        raise Exception('key %s unknown' % fkey)
    return '%s/%s' % (rgz_data_url, rgz_dn_dict[fkey])

def check_req():
    cmd = 'wget --help'
    status, msg = subprocess.getstatusoutput(cmd)
    if (status != 0):
        raise Exception('wget is not installed properly')

def download_file(url, tgt_dir):
    if (not osp.exists(tgt_dir)):
        raise Exception("tgt_dir %s not found" % tgt_dir)
    fn = url.split('/')[-1] # hack hack hack
    full_fn = osp.join(tgt_dir, fn)
    if (osp.exists(full_fn)):
        print("%s exists already, skip downloading" % full_fn)
        return full_fn
    cmd = 'wget -O %s %s' % (full_fn, url)
    print("Downloading %s to %s" % (fn, tgt_dir))
    stt = time.time()
    status, msg = subprocess.getstatusoutput(cmd)
    if (status != 0):
        raise Exception("Downloading from %s failed: %s" % (url, msg))
    print("Downloading took %.3f seconds" % (time.time() - stt))
    return full_fn

def extract_file(tar_file, tgt_dir):
    if (not osp.exists(tar_file)):
        raise Exception("file not found for extraction: %s" % tar_file)
    if (not tgt_dir.endswith('/')):
        tgt_dir += '/'
    cmd = 'tar -xf %s -C %s' % (tar_file, tgt_dir)
    print("Extracting from %s to %s" % (tar_file, tgt_dir))
    stt = time.time()
    status, msg = subprocess.getstatusoutput(cmd)
    if (status != 0):
        raise Exception("fail to extract %s: %s" % (tar_file, msg))
    print("Extraction took %.3f seconds" % (time.time() - stt))

def get_rgz_root():
    # current file directory
    cfd = osp.dirname(osp.abspath(__file__))
    rgz_root = osp.abspath(osp.join(cfd, '..'))
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

def sync_annotations(rgz_root):
    """
    Sync each index file in the imagesets with its annotation file
    """
    cwd = os.getcwd()
    anno_path = osp.join(rgz_root, 'data/RGZdevkit2017/RGZ2017/Annotations')
    os.chdir(anno_path)
    #
    index_path = osp.join(rgz_root, 'data/RGZdevkit2017/RGZ2017/ImageSets/Main')
    missing_files = []
    for indf in os.listdir(index_path):
        indf = osp.join(index_path, indf)
        with open(indf, 'r') as fin:
            fids = fin.readlines()
            fids = [x.strip() for x in fids]
            for fid in fids:
                first_id = fid.split('_')[0]
                first_id_fn = '%s.xml' % first_id
                fid_fn = '%s.xml' % fid
                if (osp.exists(first_id_fn)): # mustn't be symlink
                    if (osp.exists(fid_fn)):
                        if (osp.islink(fid_fn)):
                            continue
                        else:
                            os.remove(fid_fn)
                            os.symlink(first_id_fn, fid_fn)
                    else:
                        os.symlink(first_id_fn, fid_fn)
                else:
                    if (osp.exists(fid_fn)):
                        if (osp.islink(fid_fn)):
                            os.remove(fid_fn)
                            missing_files.append(first_id)
                            print("%s missing annotation" % first_id)
                        else:
                            os.rename(fid_fn, first_id_fn)
                            os.symlink(first_id_fn, fid_fn)
                    else:
                        missing_files.append(first_id)
                        print("%s missing annotation" % first_id)
    os.chdir(cwd)
    if (len(missing_files) > 0):
        with open('missing_first_ids', 'w') as fout:
            fout.write(os.linesep.join(missing_files))

def purge_annotations(rgz_root):
    """
    Purge unnecessary annotations
    """
    anno_path = osp.join(rgz_root, 'data/RGZdevkit2017/RGZ2017/Annotations')
    index_path = osp.join(rgz_root, 'data/RGZdevkit2017/RGZ2017/ImageSets/Main')
    first_id_set = set()
    for indf in os.listdir(index_path):
        indf = osp.join(index_path, indf)
        with open(indf, 'r') as fin:
            fids = fin.readlines()
            first_id_set = first_id_set.union(set([x.strip().split('_')[0] for x in fids]))

    cwd = os.getcwd()
    os.chdir(anno_path)
    c = 0
    print("Purging files now")
    for annof in os.listdir('.'):
        if (not annof.endswith('.xml')):
            os.remove(annof)
            continue
        first_id = annof.split('.xml')[0].split('_')[0]
        if (not first_id in first_id_set):
            os.remove(annof)
            c += 1
    os.chdir(cwd)
    print("Purged %d annotation files" % c)

def find_demo_images(rgz_root):
    """
    Find images that are neither training or testing,
    and put them into demo index
    """
    image_path = osp.join(rgz_root, 'data/RGZdevkit2017/RGZ2017/PNGImages')
    index_path = osp.join(rgz_root, 'data/RGZdevkit2017/RGZ2017/ImageSets/Main')
    first_id_set = set()
    for indf in os.listdir(index_path):
        if (not (indf.startswith('test') or indf.startswith('train'))):
            continue
        indf = osp.join(index_path, indf)
        with open(indf, 'r') as fin:
            fids = fin.readlines()
            first_id_set = first_id_set.union(set([x.strip().split('_')[0] for x in fids]))

    cwd = os.getcwd()
    os.chdir(image_path)
    demo_list = []
    print("Finding images now")
    for imgf in os.listdir('.'):
        if (not imgf.endswith('.png')):
            continue
        base_fn = imgf.split('.png')[0]
        first_id = base_fn.split('_')[0]
        if (not first_id in first_id_set):
            demo_list.append(base_fn)
    os.chdir(cwd)
    print("Found %d demo image files" % len(demo_list))
    with open(osp.join(index_path, 'demo.txt'), 'w') as fout:
        fout.write(os.linesep.join(demo_list))

def setup_png_images(rgz_root):
    png_rel = 'data/RGZdevkit2017/RGZ2017/PNGImages'
    for img_nm in ['d1_img', 'd3_img', 'd4_img']:
        setup_something(rgz_root, png_rel, img_nm)

def setup_demo_images(rgz_root):
    setup_something(rgz_root, 'data/rgzdemo', 'demo_img')

def setup_models(rgz_root):
    model_rel = 'data/pretrained_model/rgz'
    for model_nm in ['d1_model', 'd3_model', 'd4_model']:
        model_subdir = model_nm.split('_model')[0].upper() #e.g. d1_model --> D1
        setup_something(rgz_root, '%s/%s' % (model_rel, model_subdir), model_nm)

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
    setup_demo_images(rr)
    setup_models(rr)
    setup_vgg_weights(rr)
    create_empty_dirs(rr)
