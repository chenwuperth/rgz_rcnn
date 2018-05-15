#    ICRAR - International Centre for Radio Astronomy Research
#    (c) UWA - The University of Western Australia, 2017
#    Copyright by UWA (in the framework of the ICRAR)
#    All rights reserved
#
#    This library is free software; you can redistribute it and/or
#    modify it under the MIT license
#
#    Created on 15 March 2018 by chen.wu@icrar.org

import signal
import os, warnings
import os.path as osp
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import numpy as np
from astropy.io import fits
import cv2
from scipy.special import erfinv
mad2sigma = np.sqrt(2) * erfinv(2 * 0.75 - 1)

import make_contours

class TimeoutException(Exception):
    pass

def _contour_timeout_handler(signum, frame):
        print "Contour timeout {0}".format(signum)
        raise TimeoutException("Contour timeout")

def _get_contour(fits_fn, sigma_level):
    with fits.open(fits_fn) as f:
        imgdata = f[0].data
    med = np.nanmedian(imgdata)
    mad = np.nanmedian(np.abs(imgdata - med))
    sigma = mad / mad2sigma
    signal.signal(signal.SIGALRM, _contour_timeout_handler)
    signal.alarm(10)
    try:
        cs = make_contours.contour(fits_fn, sigma * sigma_level, 2.0)
    except TimeoutException as te:
        print("contour timeout on %s" % osp.basename(fits_fn))
        cs = None
    finally:
        signal.alarm(0)
    return cs

def fuse(fits_fn, ir_fn, output_dir, sigma_level=5, mask_ir=True, new_size=None,
        get_path_patch_only=False): #TODO this is dangerous
    """
    overlay radio contours on top of IR images, and
    (optionally) mask "non-related" regions with IR means

    if mask_ir is set to False, return the contour object without doing anything
    to the image
    """
    warnings.simplefilter("ignore")
    cs = _get_contour(fits_fn, sigma_level)
    warnings.simplefilter("default")
    if (cs is None):
        print("Fail to produce contour on FITS file: %s" % fits_fn)
        return None
    if (mask_ir):
        im_ir = get_masked_ir(fits_fn, None, ir_fn, output_dir, cs,
                           sigma_level=sigma_level, replace_mask_with_mean=True)
    else:
        if (type(ir_fn) == str):
            im_ir = cv2.imread(ir_fn)
        else:
            im_ir = ir_fn
    if (new_size is not None and (new_size != im_ir.shape[0])):
        print("resizing im_ir to {0}".format(new_size))
        im_ir = cv2.resize(im_ir, (new_size, new_size),
                           interpolation=cv2.INTER_LINEAR)
    w, h, d = im_ir.shape
    xsize_pix = w
    ysize_pix = h
    cs['contours'] = map(make_contours.points_to_dict, cs['contours'])
    sf_x = float(xsize_pix) / cs['width']
    sf_y = float(ysize_pix) / cs['height']
    #print(sf_x, sf_y)
    verts_all = []
    codes_all = []
    components = cs['contours']

    # Scale contours to heatmap size
    for comp in components:
        for idx,level in enumerate(comp):
            verts = [((p['x']) * sf_x, (p['y'] - 1.0) * sf_y) for p in level['arr']]
            #print(verts)
            codes = np.ones(len(verts),int) * Path.LINETO
            codes[0] = Path.MOVETO
            verts_all.extend(verts)
            codes_all.extend(codes)

    # Create matplotlib path for contours
    path = Path(verts_all, codes_all)
    ecolor = 'blue' if mask_ir else 'limegreen'
    linew = 0.25 if mask_ir else 0.25
    if (new_size is not None):
        linew = 0.35
    patch_contour = patches.PathPatch(path, facecolor='none',
                                      edgecolor=ecolor, lw=linew) #limegreen
    if (not mask_ir and get_path_patch_only):
        return patch_contour

    my_dpi = 150.0
    fig = plt.figure()
    fig.set_size_inches(xsize_pix / my_dpi, ysize_pix / my_dpi)
    #print("xsize_pix / my_dpi, ysize_pix / my_dpi", xsize_pix / my_dpi, ysize_pix / my_dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.set_xlim([0, xsize_pix])
    ax.set_ylim([ysize_pix, 0])
    ax.set_aspect('equal')
    ax.imshow(im_ir[:, :, (2, 1, 0)], origin='upper')
    ax.add_patch(patch_contour)

    # Save hard copy of the figure
    suffix = '_infraredctmask' if mask_ir else '_infraredct_det'
    output_fn = '%s' % osp.basename(ir_fn).replace('_infrared', suffix)
    output_fp = osp.join(output_dir, output_fn)
    plt.savefig(output_fp, dpi=my_dpi)
    plt.close()
    return output_fp

def get_masked_ir(fits_fn, radio_fn, ir_fn, output_dir, cs,
                  sigma_level=7, replace_mask_with_mean=False):
    """
    Mask IR images based on radio contours, which collectively form a
    convex hull mask to crop relevant area on the IR image
    """
    im_ir = cv2.imread(ir_fn)
    mask_ir = np.zeros(im_ir.shape, np.uint8)

    # mean for each channel
    bg_ir = np.ones(im_ir.shape, np.uint8)
    bg_ir[:,:,0] *= 14
    bg_ir[:,:,1] *= 45
    bg_ir[:,:,2] *= 213

    components = map(make_contours.points_to_dict, cs['contours'])
    ct_pts = np.zeros((len(components) * 4, 1, 2), dtype=np.int32)

    for i, ct in enumerate(components):
        bbox = ct[0]['bbox']
        ct_pts[4 * i + 0, 0, 0] = bbox.min_x
        ct_pts[4 * i + 0, 0, 1] = bbox.min_y
        ct_pts[4 * i + 1, 0, 0] = bbox.min_x
        ct_pts[4 * i + 1, 0, 1] = bbox.max_y
        ct_pts[4 * i + 2, 0, 0] = bbox.max_x
        ct_pts[4 * i + 2, 0, 1] = bbox.min_y
        ct_pts[4 * i + 3, 0, 0] = bbox.max_x
        ct_pts[4 * i + 3, 0, 1] = bbox.max_y

    hull = cv2.convexHull(ct_pts)
    a, b, c = hull.shape
    roi_corners = np.reshape(hull, (b, a, c))
    channel_count = 3  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask_ir, roi_corners, ignore_mask_color)
    if (replace_mask_with_mean):
        # get first masked value (foreground)
        masked_image = np.zeros(im_ir.shape, np.uint8)
        for i in range(channel_count):
            fg = cv2.bitwise_or(im_ir[:,:,i], im_ir[:,:,i], mask=mask_ir[:,:,i])
            # get second masked value (background) mask must be inverted
            mask = cv2.bitwise_not(mask_ir[:,:,i])
            bk = cv2.bitwise_or(bg_ir[:,:,i], bg_ir[:,:,i], mask=mask)
            masked_image[:,:,i] = cv2.bitwise_or(fg, bk)
    else:
        masked_image = cv2.bitwise_and(im_ir, mask_ir)
        #print("masked_image = ", masked_image.shape)
    return masked_image
