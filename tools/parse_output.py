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
import numpy as np

import astropy.io.fits as pyfits
import astropy.wcs as pywcs

def convert_box2sky(detpath, fitsdir, outpath, threshold=0.8):
    """
    Convert output of ClaRAN (boxes) into sky coordinates
    """
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

def _convert_source2box(source):
    """
    for each component C in the catalog, this function outputs two things:

        1. The name of the split file S to which, component C belongs
        2. Within S, the location of C in pixel coordinates (x1, y1, x2, y2)
    
    Note that the pixel cooridnates start from top left (rather than bottom left)
    This is used to compared the ClaRAN box 

    source:   a list of components inside a source
    """
    pass

def _get_fits_mbr(fin, row_ignore_factor=10):
	"""
	step 1: get the MBR on the 2D plane
	step 2: get the MBR on the sphere

	"""
	#print "Getting the MBR of {0}".format(fin)
	hdulist = pyfits.open(fin)
	data = hdulist[0].data
	#data[np.where(data == 0.)] = np.nan
	#hdulist1 = pyfits.open(fin)
	wcs = pywcs.WCS(hdulist[0].header)

	"""
	MBR on 2D plane is quite different from MBR on the sphere.
	e.g. pixel (imin,jmin) may not be the RA_min or DEC_min
	likewise, pixel (imax,jmax) may not be the RA_max or DEC_max on the sphere
	use the following algorithm to find RA_min/max and DEC_min/max

	1. go thru each "row" (actually col?) of the shrinked image
		1.1 get the RA of the leftmost/ rightmost pixel that is not zero
		1.2 normalise rightmost RA (in case cross the 360/0 degree)
		1.3 update the global min / max RA
	2. go thru each "col" of the shrinked image
		2.1 get the DEC of topmost / bottommost pixel that is not zero
		2.2 update the global min / max DEC
	"""
	RA_min = 10000
	RA_max = -10000
	DEC_min = 10000
	DEC_max = -10000

	width = data.shape[1]
	height = data.shape[0]

	for j in xrange(height):
		row = data[j, :]
		indices = np.where(~np.isnan(row))[0]
        if (len(indices) > 2):
            left = [indices[0], j, 0, 0] # the first index (min)
            right = [indices[-1], j, 0, 0] # the last index (max)
            sky_lr = []
            sky_lr.append(wcs.wcs_pix2world([left], 0)[0][0:2])
            sky_lr.append(wcs.wcs_pix2world([right], 0)[0][0:2])
            if (sky_lr[1][0] > sky_lr[0][0]):
                sky_lr[1][0] -= 360.0
            if (sky_lr[1][0] < RA_min):
                RA_min = sky_lr[1][0]
            if (sky_lr[0][0] > RA_max):
                RA_max = sky_lr[0][0]
	for i in xrange(width):
		col = data[:, i]
		indices = np.where(~np.isnan(col))[0]
		if (0 == len(indices)):
			continue
		top = [i, indices[-1], 0, 0]
		bottom = [i, indices[0], 0, 0]
        sky_tb = []
        sky_tb.append(wcs.wcs_pix2world([top], 0)[0][0:2])
        sky_tb.append(wcs.wcs_pix2world([bottom], 0)[0][0:2])
        if (sky_tb[1][1] < DEC_min):
            DEC_min = sky_tb[1][1]
        if (sky_tb[0][1] > DEC_max):
			DEC_max = sky_tb[0][1]

	if (RA_min < 0):
		RA_min += 360
	if (RA_max < 0):
		RA_max += 360

	#http://pgsphere.projects.pgfoundry.org/types.html
	# produce the polygon
	sqlStr = "SELECT sbox '((%10fd, %10fd), (%10fd, %10fd))'" % (RA_min, DEC_min, RA_max, DEC_max)
	#print sqlStr
	return sqlStr

def _setup_db_pool():
    from psycopg2.pool import ThreadedConnectionPool
    return ThreadedConnectionPool(1, 3, database='chen', user='chen')

def build_fits_cutout_index(fits_cutout_dir,
                            prefix='gama_linmos_corrected_clipped',
                            tablename='onedegree'):
    g_db_pool = _setup_db_pool()
    conn = g_db_pool.getconn()
    for fn in os.listdir(fits_cutout_dir):
        if (not fn.startswith(prefix)):
            continue
        if (not fn.endswith('.fits')):
            continue
        if (fn.find('-') < 0):
            continue
        fits_fn = osp.join(fits_cutout_dir, fn)
        sqlStr = _get_fits_mbr(fits_fn)
        cur = conn.cursor()
        cur.execute(sqlStr)
        res = cur.fetchall()
        if (not res or len(res) == 0):
            errMsg = "fail to calculate sbox {0}".format(sqlStr)
            print(errMsg)
            raise Exception(errMsg)
        coverage = res[0][0]
        sqlStr = """INSERT INTO {0}(coverage,fileid) VALUES('{1}','{2}')"""
        sqlStr = sqlStr.format(tablename, coverage, fn)
        print(sqlStr)
        cur.execute(sqlStr)
        conn.commit()
    g_db_pool.putconn(conn)


def convert_sky2box(catalog_csv_file, split_fits_dir):
    """
    1. work out which fits file each record belong to
    """

    # build out the fits header cache to handle queries like:
    # does this point inside this fits file?

    with open(catalog_csv_file, 'r') as fin:
        cpnlist = fin.read().splitlines()
    
    last_sid = -1
    last_source = []
    for cpnline in cpnlist[1:]:
        cpn = cpnline.split(',')
        sid = cpn[0]
        # assuming the caltalog is sorted based on the source id (island id)
        if (last_sid != sid):
            _convert_source2box(last_source)
            # more here
        else:
            last_source.append(cpn)

if __name__ == '__main__':
    """ detpath = "/Users/chen/gitrepos/ml/rgz_rcnn/data/RGZdevkit2017/results"\
    "/RGZ2017/pleiades/comp4_det_testD4_2_3.txt"

    fitsdir = '/Users/chen/gitrepos/ml/rgz_rcnn/data/RGZdevkit2017/RGZ2017/FITSImages'
    convert_box2sky(detpath, fitsdir, '/tmp') """
    fits_fn = '/Users/chen/gitrepos/ml/rgz_rcnn/data/EMU_GAMA23/split_fits/' + \
              '1deg/gama_linmos_corrected_clipped4-0.fits'
    fits_fn_path = fits_fn.replace('/gama_linmos_corrected_clipped4-0.fits', '')
    build_fits_cutout_index(fits_fn_path)
