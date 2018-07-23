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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#import ephem

#getfits_exec = '/Users/Chen/Downloads/wcstools-3.9.5/bin/getfits'
#cutout_cmd = '{0} -sv -o %s -d %s %s %s %s J2000 %d %d'.format(getfits_exec)

#subimg_exec = '/Users/Chen/proj/Montage_v3.3/Montage/mSubimage -d' #degree
subimg_exec = '/Users/chen/Downloads/Montage/bin/mSubimage'
subimg_cmd = '{0} %s %s %.4f %.4f %.4f %.4f'.format(subimg_exec)
splitimg_cmd = '{0} -p %s %s %d %d %d %d'.format(subimg_exec)
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
    clipped_fname = osp.join(work_dir, fid)
    cmd = subimg_cmd % (fname, clipped_fname, ra, dec, width, height)
    print(cmd)
    return clipped_fname

def clip_file(fname):
    """
    remove all the NaN cells surrounding the image
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

def split_file(fname, width_ratio, height_ratio, halo_ratio=50,
               show_split_scheme=False, work_dir='/tmp'):
    """
    width_ratio = current_width / new_width, integer
    height_ratio = current_height / new_height, integer
    halo in pixel
    """
    file = pyfits.open(fname)
    d = file[0].data
    fhead = file[0].header
    h = d.shape[-2] #y
    w = d.shape[-1] #x
    print(h, w)
    new_h = int(h / height_ratio)
    extra_h = h % height_ratio
    halo_h = new_h / halo_ratio
    #print(extra_h)
    ny = np.arange(height_ratio) * new_h# + (new_h / 2)
    #ny[-1] += extra_h / 2
    print(ny)
    new_w = int(w / width_ratio)
    extra_w = w % width_ratio
    halo_w = new_w / halo_ratio
    #print(extra_w)
    nx = np.arange(width_ratio) * new_w# + (new_w / 2)
    #nx[-1] += extra_w / 2
    print(nx)
    # xx, yy = np.meshgrid(nx, ny)
    # print(xx[1])
    # print(yy[1])
    if (show_split_scheme):
        fig, ax = plt.subplots(1)
        ax.imshow(np.reshape(d, [d.shape[-2], d.shape[-1]]))

    for i, x in enumerate(nx):
        for j, y in enumerate(ny):
            x1 = max(x - halo_w, 0)
            y1 = max(y - halo_w, 0)
            wd = new_w
            hd = new_h
            if (i == len(nx) - 1):
                x2 = w - 1
            else:
                x2 = min(x1 + wd + halo_w, w - 1)
            if (j == len(ny) - 1):
                y2 = h - 1
            else:
                y2 = min(y1 + hd + halo_h, h - 1)
            fid = osp.basename(fname).replace('.fits', '%d-%d.fits' % (i, j))
            #cmd = cutout_cmd % (fid, work_dir, fname, ra0, dec0, width, height)
            out_fname = osp.join(work_dir, fid)
            print(splitimg_cmd % (fname, out_fname, x1, y1, (x2 - x1), (y2 - y1)))
            #print(x1, y1, ' ', x2, y2)
            if (show_split_scheme):
                rect = patches.Rectangle((x1, y1), (x2 - x1), (y2 - y1),
                                        linewidth=1, edgecolor='r',
                                        facecolor='none')
                # Add the patch to the Axes
                ax.add_patch(rect)
    if (show_split_scheme):
        plt.show()
        plt.savefig('test.png')

if __name__ == '__main__':
    #root_dir = '/Users/Chen/proj/rgz-ml/data/EMU_GAMA23'
    root_dir = '/Users/chen/gitrepos/ml/rgz_rcnn/data/EMU_GAMA23'
    # fname = osp.join(root_dir, 'gama_linmos_corrected.fits')
    # clip_file(fname)

    fname = osp.join(root_dir, 'gama_linmos_corrected_clipped.fits')
    split_file(fname, 12, 12)
