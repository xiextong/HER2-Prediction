from keras import backend as K
from all_index_bin import *
import tensorflow as tf
from binary_crossentropy_weight import *
############################################################
import numpy as np
from functools import partial, update_wrapper
import os
from pathlib import Path
import scipy.io as sio
import scipy.io as scio
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.pyplot as plt


seg_thre = 0.5

def EuclideanLoss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    loss = K.sum(K.square(y_true_f - y_pred_f))
    return loss

def EuclideanLossWithWeight(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y = K.abs(y_true_f - y_pred_f)
    all_one = K.ones_like(y_true_f)
    y1_1 = K.clip(y - 0.15 * all_one, -1, 0)
    y1_sign = K.clip(-1 * K.sign(y1_1), 0, 1)
    y1 = y1_sign * y

    y2_1 = K.clip(y - 0.15 * all_one, 0, 5)
    y2_2 = K.clip(y2_1 - 0.5 * all_one, -1, 0)
    y2_sign = K.clip(-1 * K.sign(y2_2), 0, 1)
    y2 = y2_sign * y

    y3_1 = K.clip(y - 0.5 * all_one, 0, 5)
    y3_2 = K.clip(y3_1 - 0.8 * all_one, -1, 0)
    y3_sign = K.clip(-1 * K.sign(y3_2), 0, 1)
    y3 = y3_sign * y

    y4_1 = K.sign(y - 0.8 * all_one)
    y4_sign = K.clip(y4_1, 0, 1)
    y4 = y4_sign * y

    y_final = 0.6*y1 + 1 * y2 + 1.2 * y3 + 1.4 * y4

    loss = K.sum(K.square(y_final))

    return loss

def EuclideanLossminu(y_true, y_pred):
    print(y_true)
    loss = K.sum(K.abs(y_true - y_pred))
    return loss

def DiceCoefLoss(y_true,y_pred):

    return -DiceCoef(y_true, y_pred)

def binary_crossentropy_bin(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_gt_point = K.sum(y_true_f)
    y_bg_point = K.sum((K.ones_like(y_true_f)-y_true_f))
    y_pred_gt = y_pred_f*y_true_f+K.ones_like(y_true_f)*0.0000001
    y_pred_bg = (y_true_f-K.ones_like(y_true_f)*0.0000001)+y_pred_f*(K.ones_like(y_true_f)-y_true_f)
    gt_c = (K.sum(K.binary_crossentropy(y_true_f,y_pred_gt)))/(y_gt_point+1)
    bg_c = (K.sum(K.binary_crossentropy(y_true_f,y_pred_bg)))/(y_bg_point+1)
    bcb = 0.75*gt_c+0.25*bg_c
    return bcb

def crossentropy_0point5(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    thesh = K.ones_like(y_pred_f)
    thesh00 = K.zeros_like(y_pred_f)
    thesh05 = thesh * 0.5
    y_pred_f_05 = y_pred_f-thesh05
    y_pred_f_05 = K.maximum(y_pred_f_05,thesh00)
    y_pred_f_05 = y_pred_f_05+thesh05
    y_05_equal = K.equal(y_pred_f_05,thesh05)
    for n in range():
        y_05_equal
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def binary_crossentropy_sum(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    print(y_true_f)
    bc = (K.sum(K.binary_crossentropy(y_true_f,y_pred_f)))/262144
    return bc

def focal_loss(gamma=2., alpha=.25):

    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed

def focal_loss_fixed2(y_true, y_pred):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(0.25 * K.pow(1. - pt_1, 2) * K.log(pt_1+0.00001))-K.sum((1-0.25) * K.pow( pt_0, 2) * K.log(1. - pt_0+0.00001))

def focal_loss_weight(y_true, y_pred):
    print('y_true: ', y_true.shape)
    print('y_pred: ', y_pred.shape)
    y_true_weight = tf.expand_dims(y_true[:,:,:,0], -1)
    pt_1 = tf.where(tf.equal(y_true_weight, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true_weight, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(0.25 * K.pow(1. - pt_1, 2) * K.log(pt_1+0.00001))-K.sum((1-0.25) * K.pow( pt_0, 2) * K.log(1. - pt_0+0.00001))



def lsaf_loss(y_true,y_pred):
    #focal loss
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    f_loss = -K.sum(0.25 * K.pow(1. - pt_1, 2) * K.log(pt_1 + 0.00001)) - K.sum((1 - 0.25) * K.pow(pt_0, 2) * K.log(1. - pt_0 + 0.00001))
    #level set loss
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    phi = y_pred_f - 0.5
    Heaviside = 0.5*(1+K.tanh(20*phi))
    c1 = K.sum(y_true_f*Heaviside)/K.sum(Heaviside)
    c2 = K.sum(y_true_f*(1-Heaviside))/K.sum(1-Heaviside)
    ls_loss = K.sum(((y_true_f - c1)**2)*Heaviside) + K.sum(((y_true_f - c2)**2)*(1-Heaviside))
    return f_loss + 4*0.0001*ls_loss






























