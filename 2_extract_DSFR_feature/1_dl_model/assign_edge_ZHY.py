import cv2
import copy
import h5py
import numpy as np


def extract_contours(threshold_roi):
    edgeImg = np.zeros_like(threshold_roi)
    _, contours, _ = cv2.findContours(copy.deepcopy(threshold_roi), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        contours_npy = np.reshape(np.array(contours[i]), (-1, 2))
        edgeImg[contours_npy[:, 1], contours_npy[:, 0]] = 1
        edgeImg = np.array(edgeImg,np.uint8)
    return edgeImg