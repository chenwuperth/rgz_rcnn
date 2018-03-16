# copeied from the RGZ git hub:
#
# https://github.com/zooniverse/Radio-Galaxy-Zoo/blob/master/scripts/make_contours.py

import os
import sys
import glob
import json
import numpy as np
from astropy.io import fits
from collections import namedtuple
import cmath

EPSILON = 1e-10

NLEVELS = 13
#LEVEL_SCALE = np.sqrt(5)
THRESHOLD = 8


def make_levels(NLEVELS, LEVEL_INTERVAL):

    LEVELS = [9. / np.sqrt(3)]    # Starting value for the level scaling
    for l in range(NLEVELS):
        LEVELS.append(LEVELS[-1] * LEVEL_INTERVAL)

    return LEVELS


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return str(self.to_pair())

    def __eq__(self, b):
        x = self.x - b.x
        y = self.y - b.y
        return (x * x) + (y * y) < EPSILON

    def to_pair(self):
        return (self.x, self.y)

    def to_dict(self):
        return {"x": self.x, "y": self.y}


class Node:
    def __init__(self, parent, car, cdr=None, prev=None):
        self.parent = parent
        self.car = car
        self.cdr = cdr
        self.prev = prev

    def swap(self):
        n = self.cdr
        self.cdr = self.prev
        self.prev = n
        return n

    def remove(self):
        if self.parent.head is self:
            self.parent.head = self.cdr
        if self.parent.tail is self:
            self.parent.tail = self.prev
        if not self.prev is None:
            self.prev.cdr = self.cdr
        if not self.cdr is None:
            self.cdr.prev = self.prev
        self.prev = None
        self.cdr = None


class List:
    def __init__(self):
        self.head = None
        self.tail = None
        self.closed = False

    def __str__(self):
        s = self.head
        sr = "("
        while not s.cdr is None:
            sr = sr + str(s.car) + ", "
            s = s.cdr
        return sr + str(s.car) + ")"

    def prepend(self, value):
        node = Node(self, value, cdr=self.head)
        if not self.head is None:
            self.head.prev = node
        self.head = node
        if self.tail is None:
            self.tail = self.head
        return self

    def close(self, value):
        node = Node(self, value, prev=self.tail, cdr=self.head)
        self.head = node
        self.tail = node
        self.closed = True
        return self

    def append(self, value):
        node = Node(self, value, prev=self.tail)
        if not self.head is None:
            self.tail.cdr = node
        self.tail = node
        if self.head is None:
            self.head = self.tail
        return self

    def reverse(self):
        node = self.head
        while not node is None:
            node = node.swap()
        node = self.head
        self.head = self.tail
        self.tail = node
        return self

    def merge(self, mergee):
        self.tail.cdr = mergee.head
        mergee.head.prev = self.tail
        self.tail = mergee.tail
        return self

    def first(self):
        return self.head.car

    def last(self):
        return self.tail.car


class ContourBuilder:
    def __init__(self, lvl):
        self.lvl = lvl
        self.s = List()
        self.count = 0

    def add_segment(self, a, b):
        ss = self.s.head
        ma = None
        mb = None
        prepend_a = False
        prepend_b = False

        while (not ss is None):
            if ma is None:
                if a == ss.car.first():
                    ma = ss
                    prepend_a = True
                elif a == ss.car.last():
                    ma = ss
            if mb is None:
                if b == ss.car.first():
                    mb = ss
                    prepend_b = True
                elif b == ss.car.last():
                    mb = ss
            if (not mb is None) and (not ma is None):
                break
            else:
                ss = ss.cdr

        if ma is None and mb is None:
            ma = List().append(a).append(b)
            self.s.prepend(ma)
            self.count = self.count + 1
        elif mb is None:
            if prepend_a:
                ma.car.prepend(b)
            else:
                ma.car.append(b)
        elif ma is None:
            if prepend_b:
                mb.car.prepend(a)
            else:
                mb.car.append(a)
        else:
            self.count = self.count - 1
            if (ma.car is mb.car):
                ma.car.close(a)
            elif (not prepend_a) and (not prepend_b):
                ma.car.reverse()
                mb.car.merge(ma.car)
                ma.remove()
            elif prepend_a and (not prepend_b):
                mb.car.merge(ma.car)
                ma.remove()
            elif prepend_b and prepend_a:
                ma.car.reverse()
                ma.car.merge(mb.car)
                mb.remove()
            elif prepend_b and (not prepend_a):
                ma.car.merge(mb.car)
                mb.remove()

# contour is countring subrouter for rectangularily spaced data
#
# d - matrix of data to contour
# ilb, iub, jlb, jub - index bounds of data matric
# x - data matrix column coordinates
# y - data matrix row coordinates
# nc - number of contour levels
# z - contour levels in increasing order


def sect(h, xh, p1, p2):
    return ((h[p2] * xh[p1]) - (h[p1] * xh[p2])) / (h[p2] - h[p1])


def conrec(d, ilb, iub, jlb, jub, x, y, nc, z):
    h = [None] * 5
    sh = [None] * 5
    xh = [None] * 5
    yh = [None] * 5
    contours = [None] * nc

    x1 = 0.0
    x2 = 0.0
    y1 = 0.0
    y2 = 0.0

    im = [0, 1, 1, 0]
    jm = [0, 0, 1, 1]

    castab = [
        [
            [0, 0, 8], [0, 2, 5], [7, 6, 9]
        ],
        [
            [0, 3, 4], [1, 3, 1], [4, 3, 0]
        ],
        [
            [9, 6, 7], [5, 2, 0], [8, 0, 0]
        ]
    ]

    for j in range(jub - 1, jlb - 1, -1):
        for i in range(ilb, iub):
            dmin = min(min(d[i][j], d[i][j + 1]),
                       min(d[i + 1][j], d[i + 1][j + 1]))
            dmax = max(max(d[i][j], d[i][j + 1]),
                       max(d[i + 1][j], d[i + 1][j + 1]))

            if dmax >= z[0] and dmin <= z[nc - 1]:
                for k in range(0, nc):
                    if z[k] >= dmin and z[k] <= dmax:
                        for m in [4, 3, 2, 1, 0]:
                            if m > 0:
                                h[m] = d[i + im[m - 1]][j + jm[m - 1]] - z[k]
                                xh[m] = x[i + im[m - 1]]
                                yh[m] = y[j + jm[m - 1]]
                            else:
                                h[0] = 0.25 * (h[1] + h[2] + h[3] + h[4])
                                xh[0] = 0.5 * (x[i] + x[i + 1])
                                yh[0] = 0.5 * (y[j] + y[j + 1])
                            if h[m] > EPSILON:
                                sh[m] = 1
                            elif h[m] < -EPSILON:
                                sh[m] = -1
                            else:
                                sh[m] = 0
                        for m in [1, 2, 3, 4]:
                            m1 = m
                            m2 = 0
                            if m != 4:
                                m3 = m + 1
                            else:
                                m3 = 1

                            case_value = castab[sh[m1] +
                                                1][sh[m2] + 1][sh[m3] + 1]
                            if case_value != 0:
                                if case_value == 1:
                                    x1 = xh[m1]
                                    y1 = yh[m1]
                                    x2 = xh[m2]
                                    y2 = yh[m2]
                                elif case_value == 2:
                                    x1 = xh[m2]
                                    y1 = yh[m2]
                                    x2 = xh[m3]
                                    y2 = yh[m3]
                                elif case_value == 3:
                                    x1 = xh[m3]
                                    y1 = yh[m3]
                                    x2 = xh[m1]
                                    y2 = yh[m1]
                                elif case_value == 4:
                                    x1 = xh[m1]
                                    y1 = yh[m1]
                                    x2 = sect(h, xh, m2, m3)
                                    y2 = sect(h, yh, m2, m3)
                                elif case_value == 5:
                                    x1 = xh[m2]
                                    y1 = yh[m2]
                                    x2 = sect(h, xh, m3, m1)
                                    y2 = sect(h, yh, m3, m1)
                                elif case_value == 6:
                                    x1 = xh[m3]
                                    y1 = yh[m3]
                                    x2 = sect(h, xh, m1, m2)
                                    y2 = sect(h, yh, m1, m2)
                                elif case_value == 7:
                                    x1 = sect(h, xh, m1, m2)
                                    y1 = sect(h, yh, m1, m2)
                                    x2 = sect(h, xh, m2, m3)
                                    y2 = sect(h, yh, m2, m3)
                                elif case_value == 8:
                                    x1 = sect(h, xh, m2, m3)
                                    y1 = sect(h, yh, m2, m3)
                                    x2 = sect(h, xh, m3, m1)
                                    y2 = sect(h, yh, m3, m1)
                                elif case_value == 9:
                                    x1 = sect(h, xh, m3, m1)
                                    y1 = sect(h, yh, m3, m1)
                                    x2 = sect(h, xh, m1, m2)
                                    y2 = sect(h, yh, m1, m2)
                                if contours[k] is None:
                                    contours[k] = ContourBuilder(z[k])
                                contours[k].add_segment(
                                    Point(y1, x1), Point(y2, x2))
    return contours


def contour_list(contours):
    contours = filter(lambda c: not c is None, contours)
    l = []
    for k, c in enumerate(contours):
        s = c.s.head
        lvl = c.lvl
        while (not s is None):
            h = s.car.head
            l2 = {'arr': [], 'k': k, 'level': lvl}
            while (not h is None):
                l2['arr'].append(h.car)
                h = h.cdr
            l.append(l2)
            s = s.cdr
    return sorted(l, key=lambda n: n['k'])


def contour(f, rms, LEVEL_INTERVAL):
    BBox = namedtuple('BBox', 'max_x max_y min_x min_y')
    data = fits.getdata(f)[::-1]
    height = len(data)
    width = len(data[0])

    def filter_small(c):
        box = c['bbox']
        x = box.max_x - box.min_x
        y = box.max_y - box.min_y
        return cmath.sqrt(x * x + y * y).real > (THRESHOLD * (width / 301))

    def group_contours(c):
        group = []
        group.append(c)
        bbox = c['bbox']
        for sc in subcontours:
            p = sc['arr'][0]
            if bbox.max_x > p.x and bbox.min_x < p.x and bbox.max_y > p.y and bbox.min_y < p.y:
                group.append(sc)
        return group

    def bounding_box(c):
        xs = map(lambda p: p.x, c['arr'])
        ys = map(lambda p: p.y, c['arr'])
        max_x = max(xs)
        min_x = min(xs)
        max_y = max(ys)
        min_y = min(ys)
        c['bbox'] = BBox(max_x, max_y, min_x, min_y)
        return c

    idx = range(1, height + 1)
    jdx = range(1, width + 1)
    LEVELS = make_levels(NLEVELS, LEVEL_INTERVAL)
    cs = contour_list(conrec(data, 0, height - 1, 0, width - 1, idx,
                             jdx, len(LEVELS), map(lambda l: l * rms / LEVELS[0], LEVELS)))

    k0contours = map(bounding_box, filter(lambda c: c['k'] == 0, cs))
    subcontours = filter(lambda c: c['k'] != 0, cs)

    return {'height': height, 'width': width, 'contours': map(group_contours, filter(filter_small, k0contours))}


def points_to_dict(g):
    for i, c in enumerate(g):
        if not isinstance(c['arr'][0], dict):
            c['arr'] = map(lambda p: p.to_dict(), c['arr'])
        g[i] = c
    return g


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print "Usage: python make_contours.py [file] [RMS] [LEVEL_INTERVAL]"
        sys.exit()
    f = sys.argv[1]
    rms = float(sys.argv[2])
    LEVEL_INTERVAL = float(sys.argv[3])
    cs = contour(f, rms, LEVEL_INTERVAL)
    cs['contours'] = map(points_to_dict, cs['contours'])
    print json.dumps(cs)
