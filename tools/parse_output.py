#    ICRAR - International Centre for Radio Astronomy Research
#    (c) UWA - The University of Western Australia, 2017
#    Copyright by UWA (in the framework of the ICRAR)
#    All rights reserved
#
#    This library is free software; you can redistribute it and/or
#    modify it under the MIT license
#
#    Created on 1 August 2018 by chen.wu@icrar.org

import os
import os.path as osp
import math
import warnings
import csv
from collections import defaultdict

import astropy.io.fits as pyfits
import astropy.wcs as pywcs

def convert_box2sky(detpath, fitsdir, outpath, threshold=0.8):
    with open(detpath, 'r') as fin:
        mylist = fin.read().splitlines()
        for line in mylist:
            ll = line.split()
            #print(ll)
            if (float(ll[1]) < threshold):
                continue
            x1, y1_, x2, y2_ = [float(x) for x in ll[2:]]
            fname = osp.join(fitsdir, ll[0].split('_')[0] + '.fits')
            #print(fname)
            file = pyfits.open(fname)
            height, _ = file[0].data.shape
            fhead = file[0].header
            y2 = height - y1_
            y1 = height - y2_
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            warnings.simplefilter("ignore")
            w = pywcs.WCS(fhead, naxis=2)
            warnings.simplefilter("default")
            ra, dec = w.wcs_pix2world([[cx, cy]], 0)[0]
            print('%.4f,%.4f,%s' % (ra, dec, osp.basename(fname)))

if __name__ == '__main__':
    detpath = "/Users/chen/gitrepos/ml/rgz_rcnn/data/RGZdevkit2017/results"\
    "/RGZ2017/pleiades/comp4_det_testD4_2_3.txt"

    fitsdir = '/Users/chen/gitrepos/ml/rgz_rcnn/data/RGZdevkit2017/RGZ2017/FITSImages'
    convert_box2sky(detpath, fitsdir, '/tmp')