#    ICRAR - International Centre for Radio Astronomy Research
#    (c) UWA - The University of Western Australia, 2017
#    Copyright by UWA (in the framework of the ICRAR)
#    All rights reserved
#
#    This library is free software; you can redistribute it and/or
#    modify it under the MIT license
#
#    Created on 12 July 2018 by chen.wu@icrar.org
import os
import os.path as osp
import math
import warnings
import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import numpy as np
#import ephem

#getfits_exec = '/Users/Chen/Downloads/wcstools-3.9.5/bin/getfits'
#cutout_cmd = '{0} -sv -o %s -d %s %s %s %s J2000 %d %d'.format(getfits_exec)

subimg_exec = '/Users/Chen/proj/Montage_v3.3/Montage/mSubimage -d' #degree
subimg_cmd = '{0} %s %s %.4f %.4f %.4f %.4f'.format(subimg_exec)
"""
e.g.
/Users/Chen/proj/Montage_v3.3/Montage/mSubimage -d
/Users/Chen/proj/rgz-ml/data/EMU_GAMA23/gama_linmos_corrected.fits
/tmp/gama_linmos_corrected_clipped.fits 345.3774 -32.499 6.7488 6.1177
"""

def clip_nan(d, file, fname, work_dir='/tmp'):
    h = d.shape[0]
    w = d.shape[1]
    #print(w, h)
    # up and down
    x1, x2, y1, y2 = None, None, None, None
    for i in range(h):
        if (y1 is None and np.sum(np.isnan(d[i, :])) < w):
            y1 = i
        if (y2 is None and np.sum(np.isnan(d[h - i - 1, :])) < w):
            y2 = h - i - 1
        if (y1 is not None and y2 is not None):
            break

    # left and right
    for j in range(w):
        if (x1 is None and np.sum(np.isnan(d[:, j])) < h):
            x1 = j
        if (x2 is None and np.sum(np.isnan(d[:, w - j - 1])) < h):
            x2 = w - j - 1
        if (x1 is not None and x2 is not None):
            break
    #print(x1, x2, y1, y2)
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    #print(cx, cy)
    fhead = file[0].header
    warnings.simplefilter("ignore")
    w = pywcs.WCS(fhead, naxis=2)
    warnings.simplefilter("default")
    ra, dec = w.wcs_pix2world([[cx, cy]], 0)[0]
    #print(ra, dec)
    #ra0 = str(ephem.hours(ra * math.pi / 180)).split('.')[0]
    #dec0 = str(ephem.degrees(dec * math.pi / 180)).split('.')[0]
    width = abs((x2 - x1) * fhead['CDELT1'])
    height = abs((y2 - y1) * fhead['CDELT2'])
    fid = osp.basename(fname).replace('.fits', '_clipped.fits')
    #cmd = cutout_cmd % (fid, work_dir, fname, ra0, dec0, width, height)
    cmd = subimg_cmd % (fname, osp.join(work_dir, fid), ra, dec, width, height)
    print(cmd)

def split(fname, width_ratio, height_ratio):
    """
    """
    file = pyfits.open(fname)
    d = file[0].data
    dim = len(d.shape)
    if (dim > 2):
        if (d.shape[-3] > 1):
            raise Exception("cannot deal with cubes yet")
        else:
            d = np.reshape(d, [d.shape[-2], d.shape[-1]])
    clip_nan(d, file, fname)
    file.close()

if __name__ == '__main__':
    fname = '/Users/Chen/proj/rgz-ml/data/EMU_GAMA23/gama_linmos_corrected.fits'
    #fname = '/tmp/gama_linmos_corrected_clipped.fits'
    split(fname, None, None)
