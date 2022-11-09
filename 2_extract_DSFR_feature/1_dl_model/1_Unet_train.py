'''最新代码'''
'''
keras===2.3.1
tensorflow_gpu==2.2.0
scipy==1.2.1
python==3.6.0
scikit-learn==0.18.1
'''
#keras==2.1.6
#tensorflow==1.7.0
import cv2
# import tensorflow as tf
#print(tf.__version__)
import numpy as np
import h5py
import os
from PIL import Image
import math
import PIL

from all_model_bin import *
from all_index_bin import *
from all_loss_bin import *
from assign_edge_ZHY import *
import random

from openpyxl import workbook
import openpyxl
import pickle
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
from keras.backend.tensorflow_backend import set_session

# 设置现存占用比例
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


Version = 'C6_seg'

# the path control center
#數據讀取
PathH5 = r'../bc/data_set/C6/331_0.8_0.8_0.8_small_nor_slice/'
listFileH5 = []

# 數據讀取列表
all_rand={}


f = open('../bc/data_set/all_label.pkl','rb')
all_label = pickle.load(f)
seg_list = [pth_id for pth_id in all_label.keys() if all_label[pth_id]['seg']=='Y']
#读取训练集
for root, sub_dirs, filelist in os.walk(PathH5):
    for filename in filelist:
        pth_id = int(filename.split('_')[0])
        if pth_id in seg_list:
            listFileH5.append(os.path.join(root ,filename))
print(len(listFileH5))
trainset_num = len(listFileH5)    #所有分割数据h5文件
np.random.shuffle(listFileH5)   #打乱图片顺序


##############################
lr = 10e-4
height = 64
width = 64
batchnum = 6
data_c = np.zeros([batchnum,height,width,1],dtype=float)
label_c = np.zeros([batchnum,height,width,1],dtype =float)
data_c_test = np.zeros([1,height,width,1],dtype=float)
label_c_test = np.zeros([1,height,width,1],dtype=float)
b= np.zeros([height,width],dtype=float)
b1= np.zeros([1,height,width,1],dtype=float)
b2= np.zeros([1,height,width,1],dtype=float)
result_img = np.zeros([1,height,width,1],dtype=float)
result_final = np.zeros([height,width],dtype=float)
################################
#保存模型
result_save_path = os.path.join('../bc/try1009/0.8_0.8_0.8_smallimg112_nor_unet/C6_0.001/0925')
model_save_Path = os.path.join(result_save_path,'model')
model_name ="BC_model"
model_suffix = ".h5"
if os.path.exists(model_save_Path) == 0:
    os.makedirs(model_save_Path)
#TXT路徑
txt_path = os.path.join(result_save_path,"train_txt")

if os.path.exists(txt_path) == 0:
    os.makedirs(txt_path)


def EuclideanLoss(y_true,y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    loss = keras.sum(keras.square(y_true_f-y_pred_f))
    return loss

def Diceloss(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    loss = -1* (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)
    return loss


def EuclideanLossWithWeight(y_true,y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    y = keras.abs(y_true_f-y_pred_f)
    all_one = keras.ones_like(y_true_f)
    y1_1 = keras.clip(y-0.15*all_one,-1,0)
    y1_sign = k.clip(-1*k.sign(y1_1),0,1)
    y1 = y1_sign*y

    y2_1 = k.clip(y-0.15*all_one,0,5)
    y2_2 = k.clip(y-0.5*all_one,-1,0)
    y2_sign = k.clip(-1*k.sign(y2_2),0,1)
    y2 = y2_sign*y

    y3_1 = k.clip(y-0.5*all_one,0,5)
    y3_2 = k.clip(y-0.8*all_one,-1,0)
    y3_sign = k.clip(-1*k.sign(y3_2),0,1)
    y3 = y3_sign*y

    y4_1 = k.sign(y-0.8*all_one)
    y4_sign = k.clip(y4_1,0,1)
    y4 = y4_sign*y

    y_final = y1 + 1.4*y2 + 1.7*y3 +2*y4

    loss = keras.sum(keras.square(y_final))

    return loss

def binary_crossentropy_bin(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_gt_point = K.sum(y_true_f)
    y_bg_point = K.sum((K.ones_like(y_true_f)-y_true_f))
    y_pred_gt = y_pred_f*y_true_f+K.ones_like(y_true_f)*0.0000001
    y_pred_bg = (y_true_f-K.ones_like(y_true_f)*0.0000001)+y_pred_f*(K.ones_like(y_true_f)-y_true_f)
    gt_c = (K.sum(K.binary_crossentropy(y_true_f,y_pred_gt)))/(y_gt_point+1)
    bg_c = (K.sum(K.binary_crossentropy(y_true_f,y_pred_bg)))/(y_bg_point+1)
    bcb = 0.1*gt_c+0.9*bg_c
    return bcb



def gt_sum(y_true,y_pred):
    gt_sum = K.sum(K.flatten(y_true))
    return gt_sum



if __name__ == "__main__":

 ##train_network
    liuzhou =False
    model = twoDNet((64, 64))
    model.compile(optimizer=Adam(lr=lr), loss=focal_loss_fixed2,metrics=['acc', precision, recall, fmeasure, tp, fp, tn, fn, yt_sum])

    # iteration counter
    num_count = 0
    batch_count = 0

 # 训练过程的相关信息
    txt = open(txt_path+'TotalResult_0801.txt', 'w')
 #  创建Excel表并写入数据
    exc_name = os.path.join(txt_path, 'train_result.xlsx')
    exc_name2 = os.path.join(txt_path, 'train_result_slice.xlsx')
    ws = []  # 全局工作表对象
    wb = workbook.Workbook()  # 创建Excel对象
    ws = wb.active  # 获取当前正在操作的表对象
    ws.append(['epoch', 'recall', 'precision', ' dsc', ' loss'])
    wb.save(exc_name)
    ws1 = []  # 全局工作表对象
    wb1 = workbook.Workbook()  # 创建Excel对象
    ws1 = wb1.active  # 获取当前正在操作的表对象
    ws1.append(['epoch', ' loss'])
    wb1.save(exc_name2)

    # leave-one evaluate
    for epoch in range(500):
        num_count = 0
        print(str(epoch))
        tp_all1 = 0
        fp_all1 = 0
        tn_all1 = 0
        fn_all1 = 0
        num = 0
        for num in range(trainset_num):
            # 从h5文件名中提取到数据存放的数据文件夹
            f = h5py.File(listFileH5[num%trainset_num])
            data = f['image'][:]   #读取h5文件的数据，类似字典的键值
            label = f['label'][:]
            ####################把label换成背景
            if liuzhou:
                label = 1-label
            #print(data.shape)
            # data = f['img_A'][:]   #读取h5文件的数据，类似字典的键值
            # label = f['gt'][:]
            f.close()
            if np.sum(label) == 0:
                 continue
            # PIL.Image 转 numpy.array
            # array_img = np.array(Img)
            _, label = cv2.threshold(label, 0.5, 1, cv2.THRESH_BINARY)#二值化
            data_c[batch_count, :, :, 0] = data[:, :]
            label_c[batch_count, :, :, 0] = label[:, :]
            batch_count = batch_count + 1
            if batch_count < batchnum:
                continue
            batch_count = 0
            result_data = model.train_on_batch([data_c], label_c)
            num_count = num_count + batchnum
            print("epoch",epoch,"iteration ", num_count, " loss: \n", result_data)
            txt.write('epoch'+str(epoch)+':'+str(num_count) + ':' + str(result_data) + '  \n')
            tp_all1 = tp_all1 + result_data[5]
            fp_all1 = fp_all1 + result_data[6]
            tn_all1 = tn_all1 + result_data[7]
            fn_all1 = fn_all1 + result_data[8]

            ws1.append([epoch, result_data[0]])
            wb1.save(exc_name2)

        if epoch%1 == 0:
            str_num = str(epoch)
            model.save(os.path.join(model_save_Path,model_name + '-' + str_num + model_suffix))  # 保存模型

            recall1 = tp_all1 / (tp_all1 + fn_all1+1e-6)
            precision1 = tp_all1 / (tp_all1 + fp_all1+1e-6)
            dsc1 = (2 * recall1 * precision1) / (recall1 + precision1+1e-6)
            print('recall:' + str(recall1) + ' precision:' + str(precision1) + ' dsc:' + str(dsc1) + '\n')
            txt.write('recall:' + str(recall1) + ' precision:' + str(precision1) + ' dsc:' + str(dsc1) + '\n')
            ws.append([epoch, recall1, precision1, dsc1,result_data[0]])
            wb.save(exc_name)
        #
        recall1 = tp_all1 / (tp_all1 + fn_all1)
        precision1 = tp_all1 / (tp_all1 + fp_all1)
        dsc1 = (2 * recall1 * precision1) / (recall1 + precision1)
        print('recall:' + str(recall1) + ' precision:' + str(precision1) + ' dsc:' + str(dsc1) + '\n')
        txt.write('epoch:' + str(epoch)+'recall:' + str(recall1) + ' precision:' + str(precision1) + ' dsc:' + str(dsc1) + '\n'+ '\n')