import cv2
import numpy as np
import h5py
from matplotlib import pyplot as plt
import os
from PIL import Image
import math
import PIL
import imutils
from keras.layers import *
from keras.losses import *
from keras.optimizers import *
from keras.utils import plot_model
from keras.models import *
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,TensorBoard
from keras import backend as keras
from keras.utils.vis_utils import plot_model
from keras import backend as k
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

import random
Kfold = '1'
Version = '60_patients'+'_Kfold_'+Kfold

PathH5 = '/root/Desktop/xxt/Pancreas_segmentation_data/h5_big2/'
listFileH5 =[]


#TXT路徑
txt_name = '/root/Desktop/xxt/new/train_txt/test'
#txt_path = '/media/root/003HardDisk/YX_small_seg_256_256/'+Version+'/train_txt/'
if os.path.exists(txt_name) == 0:
    os.makedirs(txt_name)
# 數據讀取列表
all_rand=[]
#root:/root/Desktop/xxt/Pancreas_segmentation_data/h5_big2/28
#sub_dirs:['15', '39', '96', '04', '20', '29', '33', '49', '30', '38', '48', '88', '52',...
#  '01', '83', '16', '31', '64', '93', '94', '34', '24', '89', '35', '58', '22', '86', '18', '70', '40', '51', '27', '79', '62', '10', '43', '47', '44', '84', '46', '14', '45', '41', '42', '60', '61', '25', '37', '26', '68', '28', '03', '13', '17']
#filelist:'136.h5'
if __name__ == "__main__":

 # 训练过程的相关信息
    txt = open(txt_name, 'w')
    standard = 12
    n = 1
    time = 1
    for root, sub_dirs, filelist in os.walk(PathH5):
        if standard == 12:
            if time != 1:
                txt.write('\n')
            lab = 'group' + str(n)
            txt.write(lab + ':')
            # listFileH5.append(lab)
            n = n + 1
            standard = 1
        if time != 1:
            txt.write(root+';')
            standard = standard + 1
        if time == 1:
            time = time + 1

