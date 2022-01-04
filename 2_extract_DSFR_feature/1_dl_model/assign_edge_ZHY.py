# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : preprocess.py
# @Author   : zhang hongyuan
# @Date     : 2019/1/21
# @Location : Lab of Huang


import cv2
import copy
import h5py
import numpy as np


# strName = 'C:/Users/Windows10/Desktop/11-32.mat'
#
# # Load the Mat image data
# load_data = h5py.File(strName,'r')['img1']
# original_data = np.array(load_data).T
#
# # Threshold the Mat image data
# _, thresh_data = cv2.threshold(original_data, 0.12, 1, cv2.THRESH_BINARY)
# edgeImg = np.zeros_like(thresh_data)
#
# # Show the threshhold image, cause cv2.imshow() should multiply 255
# cv2.imshow('thresh_data', thresh_data*255)
# cv2.waitKey()
#
# # Check and counters image data edge, but the function of `cv2.findContours` will change the original data
# _, contours, _= cv2.findContours(copy.deepcopy(thresh_data),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
# print(type(contours), len(contours))
#
# print(np.shape(contours[0]))
#
# # Transfer to ndarray type
# for i in range(len(contours)):
#     contours_npy = np.reshape(np.array(contours[i]), (-1,2))
#     edgeImg[contours_npy[:, 1], contours_npy[:, 0]] = 1
#
# # Show the image with drawing edge
# cv2.imshow('edge_data', edgeImg*255)
# cv2.waitKey()

def extract_contours(threshold_roi):
    edgeImg = np.zeros_like(threshold_roi)
    _, contours, _ = cv2.findContours(copy.deepcopy(threshold_roi), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        contours_npy = np.reshape(np.array(contours[i]), (-1, 2))
        edgeImg[contours_npy[:, 1], contours_npy[:, 0]] = 1
        edgeImg = np.array(edgeImg,np.uint8)
    return edgeImg