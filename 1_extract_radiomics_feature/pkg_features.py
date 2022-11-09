import os
import numpy as np
import h5py
import scipy.io as sio
import SimpleITK as sitk
from radiomics import featureextractor as FEE
import pickle

def save_mat_by_dcm_base_sio(dcm_img, save_path_with_suffix, key):
    sio.savemat(save_path_with_suffix, {key: dcm_img})


load_root_path = r'E:\workspace\BC\nii_data\all_data\ICC2'
feature_save_path = os.path.join(r'E:\workspace\BC\experience\Feature\C6\ICC\0.8_0.8_0.8_nor\1130\inFeatures2')
if not os.path.exists(feature_save_path):
    os.makedirs(feature_save_path)
# 提取特征
#exist_id = [name.split('.')[0] for name in os.listdir(feature_save_path)]

for pth_file in os.listdir(load_root_path):
    if 'MRI' in pth_file:
        pth_name = str(int(pth_file.split('_')[0]))
        image_load_path = os.path.join(load_root_path,pth_file)
        label_load_path = os.path.join(load_root_path,pth_file.split('_')[0]+'_ROI.nii.gz')
        #print(label_load_path)
        # 检查输入
        # nrrd_label = sitk.ReadImage(label_load_path)
        # nrrd_image = sitk.GetArrayFromImage(label_load_path)
        # print(np.unique(nrrd_image))

        # 使用配置文件初始化特征抽取器
        extractor = FEE.RadiomicsFeatureExtractor('Params_labels.yaml')
        extractor.loadParams('Params_labels.yaml')
        print(pth_name)
        # 抽取特征
        result = extractor.execute(image_load_path, label_load_path)
        #print(result)

        # 保存特征
        #feature = [value for key, value in result.items() if not 'diagnostics' in key]#屏蔽了的特征会有diagnostics
        feature = [value for key, value in result.items() if not 'diagnostics' in key]
        feature_name = [key for key, value in result.items() if not 'diagnostics' in key]

        save_fold_path1 = os.path.join(feature_save_path)
        if not os.path.exists(save_fold_path1):
            os.makedirs(save_fold_path1)



        feature_message = {}
        feature_message['Features'] = np.array(feature)
        feature_message['Features_name'] = np.array(feature_name) #将数据写入文件的主键data下面

        with open(os.path.join(save_fold_path1, pth_name +'.pkl'), "wb") as f:
            # 将列表a序列化后写入文件
            pickle.dump(feature_message, f)


