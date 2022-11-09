import warnings
warnings.filterwarnings('ignore')

import os
import h5py
import numpy as np
from copy import deepcopy
from sklearn.cluster import KMeans

from scipy.stats import pearsonr
from scipy.spatial.distance import pdist

import nibabel as nib
import torch

from keras.layers import *
from keras.losses import *
from keras.optimizers import *
from keras.utils import plot_model
from keras.models import *
from keras.backend.tensorflow_backend import set_session

from all_loss_bin import *
#from all_model_bin import *
from all_model_bin import *
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuit
from all_index_bin import *
import xlwt



# 设置调用GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# nii数据读取
niilist = []
roilist = []
#3D 数据
niipath = '../bc/data_set/C6/all_0.8_0.8_0.8_small/'
#niilist = os.listdir(niipath)
for filewalks in os.walk(niipath):
    for files in filewalks[2]:
        if '_ROI' in files:

            roilist.append(files)
        else:
            niilist.append(files)
# save path
savepath = '../bc/reault/0.8_0.8_0.8_smallimg112_nor_unet/C6_0.001/ori/class_result/26/inner_feature_act10'
max_feature_save_path = os.path.join(savepath,'all_max_feature')#取最大簇的特征矩阵(平均之前)



if not os.path.exists(os.path.dirname(max_feature_save_path)):
    os.makedirs(os.path.dirname(max_feature_save_path))



def is_weight_match_model(path_weight, object_model):
    with h5py.File(path_weight, mode='r') as f:
        weight_name_list = list(f['model_weights'].keys())
    layer_name_list = [layer.name for layer in object_model.layers]
    return set(layer_name_list) - set(weight_name_list) == set(weight_name_list) - set(layer_name_list)

def define_extract_model(seg_model, weight_path, extract_layer_name, plot_save_path='Model.png'):
    # 检查模型与权重是否适配
    if not is_weight_match_model(weight_path, seg_model):
        raise ConnectionError('Weight not match with Model')

    # 配置权重, 编译模型
    seg_model.load_weights(weight_path, by_name=True)
    #seg_model.compile(optimizer=SGD(lr=1.0e-8), loss=EuclideanLoss, metrics=['acc', precision, recall, fmeasure])
    seg_model.compile(optimizer=Adam(lr=1.0e-4), loss=focal_loss_fixed2,
                  metrics=['acc', precision, recall, fmeasure, tp, fp, tn, fn, yt_sum])
    # 提取分割网络输出和高维特征
    pred_value = seg_model.output
    f_value = ((seg_model.get_layer(extract_layer_name)).output)

    # 重新建立模型
    extract_model = Model(inputs=[seg_model.input], outputs=[f_value])

    # 可视化模型结构
    plot_model(extract_model, plot_save_path, show_shapes=True)
    return extract_model


def Excel_save(data, path):
    f = xlwt.Workbook() # 创建工作簿
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True) # 创建sheet
    [h, l] = data.shape # h为行数，l为列数
    for i in range(h):
        for j in range(l):
          sheet1.write(i, j, data[i, j])
    f.save(path)





if __name__ == '__main__':

    img_w = 64
    img_h = 64
    class_num = 331

    # 读取含有病灶的特征向量 read finger with has label f-value(1 * 1024) ---

    modelnumnum = '26'   # best model num
    weight_path = r'../BC_model-' + modelnumnum + '.h5'

    # 加载模型
    unet_seg_model = twoDNet((img_w, img_h, 1))

    # 定义提取模型
    extract_model = define_extract_model(unet_seg_model, weight_path=weight_path, extract_layer_name='activation_9')
    # 读取到一名患者的所有切片特征作为一个列表 upload patient slice f-value as a list

    # build a
    feature_219_1024 = np.zeros((class_num,1024))

    for I in range(class_num):

        # 提取ｓｅｍａｎｔｉｃ　特征！
        pth_name = niilist[I].split('_')[0]
        loadpathm = niipath + niilist[I]
        MRI = nib.load(loadpathm)
        MRI_data = np.array(MRI.get_data())
        MRI_G_data = (MRI_data - np.min(MRI_data)) / (np.max(MRI_data) - np.min(MRI_data))
        #MRI_G_data = MRI_data
        image_num = MRI_data.shape[2]

        mask_f_values_list = np.zeros((1,1024,image_num,1))

        for j in range(0, image_num):

            img_batch = np.zeros((1, img_w, img_h, 1))
            seg_img_gap = np.zeros((image_num, 1, 1, 1024))
            im = MRI_G_data[:, :, j]
            img_batch[0, :, :, 0] = MRI_G_data[:, :, j]

            # semantic feature extraction
            seg_img_pred = extract_model.predict_on_batch(img_batch)#得到

            # GAP
            seg_img_pred_c = np.moveaxis(seg_img_pred, 3, 1)  # 换成channelfirst

            seg_img_pred_t = torch.FloatTensor(seg_img_pred_c)
            G = torch.nn.AdaptiveAvgPool2d((1, 1))
            GAP = G(seg_img_pred_t)
            GAP_2 = GAP.numpy()  #1024x1
            mask_f_values_list[0,:,j,0] = GAP_2.reshape((1024))   #每层都输入进去#1x1024xhx1
            print(str(loadpathm), '-', str(j + 1))


        # 将所有切片特征转换为一个矩阵
        mask_f_values = np.vstack(mask_f_values_list)#1024xhx1

        mask = np.zeros((1024,image_num))
        mask[:,:] = mask_f_values[:,:,0]
        mask = np.moveaxis(mask, 0, 1) #hx1024

        # mask = np.moveaxis(mask_f_values, 0, 1)
        # mask = np.moveaxis(mask, 2, 1)

        print(mask_f_values.shape)
        print(mask.shape)

        # 聚类分析 Kmeans f-value cluster
        n_cluster = KMeans(n_clusters=2).fit_predict(deepcopy(mask))#返回每个数据对应标签

        # 获取到聚类最大簇的标签
        max_cluster = np.argmax(np.bincount(n_cluster))#统计0/1出现的次数 返回[0的次数 1的次数]   最终返回大的值所在的索引值

        print(max_cluster, n_cluster)

        # 提取聚类两个簇对应的切片特征 Kmeans Max and Min cluster f-values
        Kmeans_max_mask_f_value = mask[n_cluster == max_cluster, :]
        Kmeans_min_mask_f_value = mask[n_cluster != max_cluster, :]

        #保存最大簇特征
        print(Kmeans_min_mask_f_value.shape)
        print(Kmeans_max_mask_f_value.shape)


        if not os.path.exists(os.path.join(max_feature_save_path)):
            os.makedirs(os.path.join(max_feature_save_path))
        with h5py.File(os.path.join(max_feature_save_path, '%s.h5' % pth_name)) as f:
            f.create_dataset('Kmeans_max_mask_f_value', data=Kmeans_max_mask_f_value)
