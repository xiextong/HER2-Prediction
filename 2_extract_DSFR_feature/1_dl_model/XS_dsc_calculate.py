import cv2

import os
from pathlib import Path
import scipy.io as scio
import h5py
import PIL
import numpy as np
import cv2
import re

import math
import operator
from functools import reduce

seg_thre = 0.2

#统计每张图像的tp/tn/fp/fn
def result_calculate(save_img,label_d):
    tp, tn, fp, fn, other = 0, 0, 0, 0, 0
    pred = save_img
    roi = label_d
    x = len(save_img)
    y = len(save_img[0])

    _, pred1 = cv2.threshold(pred, seg_thre, 1, cv2.THRESH_BINARY)

    for xx in range(x):
        for yy in range(y):
            if roi[xx][yy] == 1 and pred1[xx][yy] == 1:
                tp = tp + 1
            elif roi[xx][yy] == 1 and pred1[xx][yy] == 0:
                fn = fn + 1
            elif roi[xx][yy] == 0 and pred1[xx][yy] == 1:
                fp = fp + 1
            elif roi[xx][yy] == 0 and pred1[xx][yy] == 0:
                tn = tn + 1
            else:
                other = other + 1

    return tp,fp,tn,fn




