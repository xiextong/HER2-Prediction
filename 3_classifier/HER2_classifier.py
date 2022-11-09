import matplotlib.pyplot as plt
import h5py
import sklearn.metrics as metrics
from sklearn.model_selection import  StratifiedKFold
import os
import gc
from sklearn.linear_model import LogisticRegression
import copy
from sklearn import preprocessing
import warnings
import pandas as pd
import random
import numpy as np
import pickle
warnings.filterwarnings('ignore')

from sklearn.linear_model import Lasso
from sklearn import feature_selection


sep = os.sep
filesep = sep

task = 'label_her2'

# 设置随机数种子
random.seed(6666)

Inner_PATIENTS_Label = {}
Out_PATIENTS_Label = {}
####################得到label#############
pkl_file = open(r'E:\Workspace\Project\BC\HER2\EX\all_label.pkl', 'rb')
alldata_label = pickle.load(pkl_file)
in_p_num = 0
out_p_num = 0
for pth_id in alldata_label.keys():
    if alldata_label[pth_id]['set'] == 'center1':
        Inner_PATIENTS_Label[pth_id] = alldata_label[pth_id][task]
        if alldata_label[pth_id][task] == 1:
            in_p_num = in_p_num + 1
    elif alldata_label[pth_id]['set'] == 'center2':
        Out_PATIENTS_Label[pth_id] = alldata_label[pth_id][task]
        if alldata_label[pth_id][task] == 1:
            out_p_num = out_p_num + 1

# 在任意轴打乱顺序
def array_shuffle(x, axis=0):
    new_index = list(range(x.shape[axis]))
    random.shuffle(new_index)
    x_new = np.transpose(x, ([axis] + [i for i in list(range(len(x.shape))) if i is not axis]))
    x_new = x_new[new_index][:]
    new_dim = list(np.array(range(axis)) + 1) + [0] + list(np.array(range(len(x.shape) - axis - 1)) + axis + 1)
    x_new = np.transpose(x_new, tuple(new_dim))
    return x_new

def sigmoid_y(x, thresold=0.5):
    if x < thresold:
        x = 0
    else:
        x = 1
    return x

def getAccSenSpcAuc_accbest(label, pre, pre_bestthresold=None):
    """
    只适用于二分类
    :param label:01标签
    :param pre:属于1类的概率
    :param pre_bestthresold:可手动设置阈值，否则返回roc曲线上约登指数最大处阈值
    :return:
    """
    final_true_label = label
    final_pred_value = pre
    patient_num = len(final_true_label)

    # 计算auc，并计算最佳阈值
    if (sum(final_true_label) == patient_num) or (sum(final_true_label) == 0):
        Aucc = 0
        print('only one class')
    else:
        Aucc = metrics.roc_auc_score(final_true_label, final_pred_value)
        # print('AUC', Aucc)
    fpr, tpr, _ = metrics.roc_curve(final_true_label, final_pred_value)
    # print('AUC', Aucc)
    plt.title('out-2-ROC')
    plt.plot(fpr, tpr)
    plt.show()

    # 计算最佳阈值
    # fpr, tpr, thresholds = metrics.roc_curve(final_true_label, final_pred_value)
    # # 计算约登指数
    # Youden_index = tpr + (1 - fpr)
    # best_thresold = thresholds[Youden_index == np.max(Youden_index)][0]
    #
    # # have no idea about that threshold is bigger than 1 sometimes
    # # maybe can find in https://github.com/scikit-learn/scikit-learn/commit/4d9a67f77787ffe9955187865f9b95e19286f069
    # # or https://github.com/scikit-learn/scikit-learn/issues/3097
    # if best_thresold > 1:
    #     best_thresold = 0.5

    # best_thresold = 0.5  # 认为设定阈值，因为如果预测值很hard（偏向0或1），那么阈值也会很偏（毕竟取的是预测值），但是实际情况可能取0.5为阈值也没啥大区别

    # 如果有预设阈值，则使用预设阈值计算acc，sen，spc
    if pre_bestthresold is not None:
        best_thresold = pre_bestthresold
        # 根据最终list来计算最终指标
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for nn in range(patient_num):
            t_label = final_true_label[nn]  # true label
            p_value = final_pred_value[nn]

            p_label = sigmoid_y(p_value, best_thresold)

            if (t_label == 1) and (t_label == p_label):
                tp = tp + 1  # 真阳
            elif (t_label == 0) and (t_label == p_label):
                tn = tn + 1  # 真阴
            elif (t_label == 1) and (p_label == 0):
                fn = fn + 1  # 假阴
            elif (t_label == 0) and (p_label == 1):
                fp = fp + 1  # 假阳

        Sensitivity = tp / ((tp + fn) + (1e-16))
        Specificity = tn / ((tn + fp) + (1e-16))
        Accuracy = (tp + tn) / ((tp + tn + fp + fn) + (1e-16))
    else:
        Accuracy = 0
        for thresold in pre:
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for nn in range(patient_num):
                t_label = final_true_label[nn]  # true label
                p_value = final_pred_value[nn]

                p_label = sigmoid_y(p_value, thresold)

                if (t_label == 1) and (t_label == p_label):
                    tp = tp + 1  # 真阳
                elif (t_label == 0) and (t_label == p_label):
                    tn = tn + 1  # 真阴
                elif (t_label == 1) and (p_label == 0):
                    fn = fn + 1  # 假阴
                elif (t_label == 0) and (p_label == 1):
                    fp = fp + 1  # 假阳

            Sen = tp / ((tp + fn) + (1e-16))
            Spe = tn / ((tn + fp) + (1e-16))
            Acc = (tp + tn) / ((tp + tn + fp + fn) + (1e-16))
            if Accuracy<Acc:
                Accuracy = Acc
                best_thresold = thresold
                Sensitivity = Sen
                Specificity = Spe

    return [Accuracy, Sensitivity, Specificity, Aucc, best_thresold]

def getAccSenSpcAuc(label, pre, pre_bestthresold=None):
    """
    只适用于二分类
    :param label:01标签
    :param pre:属于1类的概率
    :param pre_bestthresold:可手动设置阈值，否则返回roc曲线上约登指数最大处阈值
    :return:
    """
    final_true_label = label
    final_pred_value = pre
    patient_num = len(final_true_label)

    # 计算auc，并计算最佳阈值
    if (sum(final_true_label) == patient_num) or (sum(final_true_label) == 0):
        Aucc = 0
        print('only one class')
    else:
        Aucc = metrics.roc_auc_score(final_true_label, final_pred_value)
        # print('AUC', Aucc)

    # 计算最佳阈值
    fpr, tpr, thresholds = metrics.roc_curve(final_true_label, final_pred_value)
    # 计算约登指数
    Youden_index = tpr + (1 - fpr)
    best_thresold = thresholds[Youden_index == np.max(Youden_index)][0]

    # have no idea about that threshold is bigger than 1 sometimes
    # maybe can find in https://github.com/scikit-learn/scikit-learn/commit/4d9a67f77787ffe9955187865f9b95e19286f069
    # or https://github.com/scikit-learn/scikit-learn/issues/3097
    if best_thresold > 1:
        best_thresold = 0.5

    # best_thresold = 0.5  # 认为设定阈值，因为如果预测值很hard（偏向0或1），那么阈值也会很偏（毕竟取的是预测值），但是实际情况可能取0.5为阈值也没啥大区别

    # 如果有预设阈值，则使用预设阈值计算acc，sen，spc
    if pre_bestthresold is not None:
        best_thresold = pre_bestthresold

    # 根据最终list来计算最终指标
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for nn in range(patient_num):
        t_label = final_true_label[nn]  # true label
        p_value = final_pred_value[nn]

        p_label = sigmoid_y(p_value, best_thresold)

        if (t_label == 1) and (t_label == p_label):
            tp = tp + 1  # 真阳
        elif (t_label == 0) and (t_label == p_label):
            tn = tn + 1  # 真阴
        elif (t_label == 1) and (p_label == 0):
            fn = fn + 1  # 假阴
        elif (t_label == 0) and (p_label == 1):
            fp = fp + 1  # 假阳

    Sensitivity = tp / ((tp + fn) + (1e-16))
    Specificity = tn / ((tn + fp) + (1e-16))
    Accuracy = (tp + tn) / ((tp + tn + fp + fn) + (1e-16))

    return [Accuracy, Sensitivity, Specificity, Aucc, best_thresold]


def zhenghe(data, distill_mode):
    """
    # 整理数据到病人单位 ，即将病人特征储存到字典里===============================================
    :param data: 字典
    :param distill_mode: 三个,max,mean,maxmean

    :return:
    """
    id_all = []
    all_key = data.keys()  # np.unique会自动升序排序
    data_dict = {}
    for id in all_key:
        # data_person = data[data[:, 0] == id, :]
        data_dict[id] = {'dsfrfeature': data[id]['dsfr_feature'], 'radiofeature': data[id]['radio_feature'],
                         'label': data[id]['label']}

    # 提前整合每个病人的特征(取均值or max，or均值+max titu1996那个方法) =======================================
    # in
    feature = []
    radiofeature = []
    label = []
    for idd in all_key:

        if distill_mode is 'max':
            # print('distill_mode is ', distill_mode)
            # data_dict[idd]['dsfrfeature'] = list(data_dict[idd]['dsfrfeature'])
            feature.append(list(np.max(data_dict[idd]['dsfrfeature'], axis=0)))  # 列
        elif distill_mode is 'mean':
            # print('distill_mode is ', distill_mode)
            feature.append(list(np.mean(data_dict[idd]['dsfrfeature'], axis=0)))
        elif distill_mode is 'maxmean':
            # print('distill_mode is ', distill_mode)
            feature.append(list(0.5 * (np.mean(data_dict[idd]['dsfrfeature'], axis=0) +
                                       np.max(data_dict[idd]['dsfrfeature'], axis=0))))
        else:
            raise Exception("distill_mode must be one of them (max,mean,maxmean) ")
        radiofeature.append(list(data_dict[idd]['radiofeature']))
        id_all.append(int(idd))
        label.append(data_dict[idd]['label'])
    dsfr_feature = np.array(feature)
    radio_feature = np.array(radiofeature)
    label = np.array(label)
    id_all = np.array(id_all)
    return [id_all, label, dsfr_feature, radio_feature]

def find_feature_lasso(Data, label, alpha,random_state):
    # alphas = np.logspace(-100, 1, 50)
    # model_lassoCV = LassoCV(alphas=alphas, cv=10,max_iter=1000).fit(Data, label)
    model_lassoCV = Lasso(alpha=alpha, max_iter=500,random_state=random_state).fit(Data, label)  # HER2 #0.0008 #a=0.001 rd = 26 AUC 0.8
    # model_lassoCV = LassoCV(cv=10,max_iter=500,n_alphas=100,n_jobs=15,tol=1e-6).fit(Data, label)
    coef = model_lassoCV.coef_
    return coef


def lasso_sf(feature,feature_out=None,xuhao=None, alpha=0.0008,random_state=None):
    #参数设定
    if feature_out is not None:
        test_outside = True
    xuhao_out = 0
    alpha = alpha
    print('select features by lasso')
    # lasso筛选特征
    coel = find_feature_lasso(feature, label, alpha,random_state=random_state)
    feature = feature[:, coel != 0]
    if test_outside:
        feature_out = feature_out[:, coel != 0]
        xuhao_out = xuhao[:, coel != 0]

    return feature, feature_out, xuhao_out

def chi_sf(feature, label, feature_out=None,xuhao=None, pers_chi=99):
    test_outside = False
    if feature_out is not None:
        test_outside = True
    function = feature_selection.chi2
    print('select features by chi2')
    fs = feature_selection.SelectPercentile(function, percentile=int(pers_chi))
    feature = fs.fit_transform(feature, label)
    if test_outside:
        feature_out = fs.transform(feature_out)
        xuhao_out = fs.transform(xuhao)
    return feature, feature_out, xuhao_out


def mir_sf(feature, label, feature_out=None,xuhao=None, pers_chi=99):
    test_outside = False
    if feature_out is not None:
        test_outside = True
    function = feature_selection.mutual_info_regression
    print('select features by mutual_info_regression')
    fs = feature_selection.SelectPercentile(function, percentile=int(pers_chi))
    feature = fs.fit_transform(feature, label)
    if test_outside:
        feature_out = fs.transform(feature_out)
        xuhao_out = fs.transform(xuhao)
    return feature, feature_out, xuhao_out

if __name__ == '__main__':
    # ===================保存路径和输入特征路径====================================================================
    root_path = r'E:\Workspace\Project\BC\HER2\EX'
    savepath = os.path.join(root_path, 'Result', 'out_result3')  # 结果保存路径
    try:
        os.makedirs(savepath)
    except:
        pass

    inner_radio_feature_path = os.path.join(root_path, r'Feature\C6\0.8_0.8_0.8_nor\1130\out2_C2')
    inner_dsfr_feature_path = os.path.join(root_path, r'Feature\C6\DL_feature\C6_nor_lr0.001\26\out2_feature_C2_nor1\all_max_feature')
    out_radio_feature_path = os.path.join(root_path, r'Feature\C6\0.8_0.8_0.8_nor\1130\60out_Feature')
    out_dsfr_feature_path = os.path.join(root_path, r'Feature\C6\DL_feature\C6_nor_lr0.001\26\out_feature_nor1\all_max_feature')

    # 其他参数 -----------------------------------------StratifiedKFold-----------------------------------------------------
    test_outside = True
    k_fold = 10  # 交叉验证折数，带平衡！！的交叉验证分折
    N = 1 # 跑N次重复试验
    state = 'grid2'

    distill_mode = 'mean'  # 病人block整合为一个向量的策略，max，mean或者maxmean
    save_prob = True  # 是否保留每次实验的预测值（if true，prob，label和id都要保存到一个txt中）

    # 显示哪种结果
    '''
    1:dsfr
    2:radiomics
    3:dsfr  radiomics分数相加
    '''
    key_word_model = ['dsfr', 'radiomics']
    show_choose = 3
    # 外部检查点=====================================================================
    # 建立个文件，代表开始run
    filename = savepath + sep + 'start_flagfile.txt'
    f = open(filename, 'a+')
    writewords = 'its running !!!!!!'
    f.write(writewords)
    f.close()

    inner_radio_feature_path_list = os.listdir(inner_radio_feature_path)
    inner_dsfr_feature_path_list = os.listdir(inner_dsfr_feature_path)
    out_radio_feature_path_list = os.listdir(out_radio_feature_path)
    out_dsfr_feature_path_list = os.listdir(out_dsfr_feature_path)

    test1 = 0
    for iiiiiiiiii in range(N):
        gc.collect()  # 释放内存
        print(iiiiiiiiii)
        for xunhuan in range(1):
            test1 = test1 + 1
            print('test:', test1, '/', N, '======================================================')
            # print(datapath)

            # 读数据 ==================================================================================================
            # 读数据step1：首先读入数据
            data = {}
            for pth_id in Inner_PATIENTS_Label.keys():
                # dsfr_feature
                H5_file_dsfr = h5py.File(os.path.join(inner_dsfr_feature_path, str(pth_id) + '.h5'), 'r')
                pth_name = int(pth_id)
                dsfr_feature = H5_file_dsfr['Kmeans_max_mask_f_value'][:]
                H5_file_dsfr.close()
                data[int(pth_name)] = {}
                data[int(pth_name)]['dsfr_feature'] = dsfr_feature
                data[int(pth_name)]['label'] = Inner_PATIENTS_Label[int(pth_name)]
                # radio_feature
                H5_file_radio = h5py.File(os.path.join(inner_radio_feature_path, str(pth_id) + '.h5'), 'r')
                if len(H5_file_radio['Features'][:].shape) == 1:
                    radio_feature = H5_file_radio['Features'][:]
                else:
                    radio_feature = H5_file_radio['Features'][:][0]
                H5_file_radio.close()
                data[int(pth_name)]['radio_feature'] = radio_feature

            # 读数据step2：读入外验证数据
            out_data = {}
            if test_outside:
                for pth_id in Out_PATIENTS_Label.keys():
                    pth_name = int(pth_id)
                    # dsfr
                    H5_file = h5py.File(os.path.join(out_dsfr_feature_path, str(pth_id) + '.h5'), 'r')
                    out_dsfr_feature = H5_file['Kmeans_max_mask_f_value'][:]
                    H5_file.close()

                    out_data[int(pth_name)] = {}
                    out_data[int(pth_name)]['dsfr_feature'] = out_dsfr_feature
                    out_data[int(pth_name)]['label'] = Out_PATIENTS_Label[int(pth_name)]
                    # radiomics

                    H5_file = h5py.File(os.path.join(out_radio_feature_path, str(pth_id) + '.h5'), 'r')
                    if len(H5_file['Features'][:].shape) == 1:
                        out_radio_feature = H5_file['Features'][:]
                    else:
                        out_radio_feature = H5_file['Features'][:][0]

                    out_data[int(pth_name)]['radio_feature'] = out_radio_feature
                    H5_file.close()

            # 整理数据到病人单位，即将病人特征储存到字典里===============================================
            id_all, label, dsfrfeature, radiofeature = zhenghe(data, distill_mode)
            print(
                '{} in data num    0:1=={}:{}'.format(len(id_all), int(len(Inner_PATIENTS_Label) - in_p_num), in_p_num))
            if test_outside:
                id_all_out, label_out, dsfrfeature_out, radiofeature_out = zhenghe(out_data, distill_mode)
                print('{} out data num    0:1=={}:{}'.format(len(id_all_out), int(len(Out_PATIENTS_Label) - out_p_num),out_p_num))

            # 特征归一化
            # 先归一化外部，不然内部被覆盖后，最值就变了
            if test_outside:  # 按照内部数据归一化
                radiofeature_out = (radiofeature_out - radiofeature.min(axis=0)) / (
                            radiofeature.max(axis=0) - radiofeature.min(axis=0) + 1e-12)
                dsfrfeature_out = (dsfrfeature_out - dsfrfeature.min(axis=0)) / (
                            dsfrfeature.max(axis=0) - dsfrfeature.min(axis=0) + 1e-12)

            dsfrfeature = (dsfrfeature - dsfrfeature.min(axis=0)) / (
                        dsfrfeature.max(axis=0) - dsfrfeature.min(axis=0) + 1e-12)
            radiofeature = (radiofeature - radiofeature.min(axis=0)) / (
                        radiofeature.max(axis=0) - radiofeature.min(axis=0) + 1e-12)

            ##################################################特征选择############################################
            # 分类器的rdstate
            radio_grid_rdstate= 2900
            dsfr_grid_rdstate= 5443
            print('radio_grid_rdstate', radio_grid_rdstate)
            print('dsfr_grid_rdstate', dsfr_grid_rdstate)

            xuhao_radio = np.array([[it for it in range(radiofeature_out.shape[1])]])
            xuhao_dsfr = np.array([[it for it in range(dsfrfeature_out.shape[1])]])

            inner_radio_feature_choose, out_radio_feature_choose,xuhao_radio = lasso_sf(radiofeature, feature_out=radiofeature_out, xuhao=xuhao_radio, alpha=0.0008533, random_state=radio_grid_rdstate)
            inner_radio_feature_choose, out_radio_feature_choose,xuhao_radio = mir_sf(inner_radio_feature_choose, label, feature_out=out_radio_feature_choose, xuhao=xuhao_radio, pers_chi=85)
            inner_dsfr_feature_choose, out_dsfr_feature_choose,xuhao_dsfr = lasso_sf(dsfrfeature, feature_out=dsfrfeature_out, xuhao=xuhao_dsfr, alpha=0.004942,random_state=dsfr_grid_rdstate)
            inner_dsfr_feature_choose, out_dsfr_feature_choose,xuhao_dsfr = chi_sf(inner_dsfr_feature_choose, label, feature_out=out_dsfr_feature_choose, xuhao=xuhao_dsfr, pers_chi=78)

            #读出可重复性强的特征
            icc_excel = pd.read_excel(r'E:\Workspace\Project\BC\HER2\EX\Result\ICC\ICC59.xlsx')
            for it,icc_feature in enumerate(icc_excel.Feature):
                inner_radio_feature_choose=np.delete(inner_radio_feature_choose,np.where(xuhao_radio[0,:]==icc_feature),axis=1)
                out_radio_feature_choose=np.delete(out_radio_feature_choose, np.where(xuhao_radio[0, :] == icc_feature), axis=1)
                xuhao_radio=np.delete(xuhao_radio, np.where(xuhao_radio[0, :] == icc_feature), axis=1)
                if it >74:
                    print(it)
                    break
            print('radio choose feature:{} \ndsfr choose feature:{} '.format(inner_radio_feature_choose.shape[1],inner_dsfr_feature_choose.shape[1]))

            # 分折
            num_subjects = inner_radio_feature_choose.shape[0]
            cvrdstate =3529
            print('cvrdstate', cvrdstate)
            sfolder = StratifiedKFold(n_splits=k_fold, random_state=cvrdstate, shuffle=True)


            # 开搞
            # 构建储存结果的容器
            label_CV = []
            id_CV = []

            dsfrprob_lr = []
            radioprob_lr = []

            radio_important = []
            dsfr_important = []

            if test_outside:
                dsfrlabel_CV_out = []
                dsfrprob_lr_out = []

                radiolabel_CV_out = []
                radioprob_lr_out = []

            # 固定参数 ============================================================================================
            final_lr_penalty = 'l2'
            final_lr_C = 1.2
            final_lr_max_iter = 100
            final_lr_tol = 1e-05

            fold_flag = 1  # 记录交叉验证跑到第几折了
            for train, test in sfolder.split(inner_dsfr_feature_choose, label):
                print('doing with fold ', fold_flag)
                print(test)

                fold_flag = fold_flag + 1
                dsfrfeature_train, label_train = [inner_dsfr_feature_choose[train], label[train]]
                radiofeature_train, label_train = [inner_radio_feature_choose[train], label[train]]

                dsfrfeature_test, label_test = [inner_dsfr_feature_choose[test], label[test]]
                radiofeature_test, label_test = [inner_radio_feature_choose[test], label[test]]

                id_CV = id_CV + list(id_all[test])
                current_fold_train_id = list(id_all[train])
                current_fold_test_id = list(id_all[test])

                if test_outside:
                    dsfrfeature_out_test = copy.deepcopy(out_dsfr_feature_choose)
                    radiofeature_out_test = copy.deepcopy(out_radio_feature_choose)
                # 标准化 # 将上述得到的scale参数应用至测试数据
                StandardScaler = preprocessing.StandardScaler()  #
                dsfrfeature_train = StandardScaler.fit_transform(dsfrfeature_train)
                dsfrfeature_test = StandardScaler.transform(dsfrfeature_test)

                if test_outside:
                    dsfrfeature_out_test = StandardScaler.transform(dsfrfeature_out_test)

                StandardScaler = preprocessing.StandardScaler()
                radiofeature_train = StandardScaler.fit_transform(radiofeature_train)
                radiofeature_test = StandardScaler.transform(radiofeature_test)

                # 外部数据
                if test_outside:
                    radiofeature_out_test = StandardScaler.transform(radiofeature_out_test)

                # Logistics ---------------------------------------------------------------------------------------
                if 'dsfr' in key_word_model:
                    clf = LogisticRegression(penalty=final_lr_penalty, C=final_lr_C, max_iter=final_lr_max_iter,
                                             tol=final_lr_tol, solver='lbfgs', warm_start=True,
                                             random_state=dsfr_grid_rdstate)

                    clf.fit(dsfrfeature_train, label_train)
                    # 预测测试集
                    y_prob = clf.predict_proba(dsfrfeature_test)
                    dsfrprob_lr = dsfrprob_lr + list(y_prob[:, 1])
                    important = clf.coef_[0]
                    dsfr_important.append(list(important))
                    # 测试外验证
                    if test_outside:
                        y_prob = clf.predict_proba(dsfrfeature_out_test)
                        dsfrprob_lr_out.append(list(y_prob[:, 1]))
                if 'radiomics' in key_word_model:
                    clf = LogisticRegression(penalty=final_lr_penalty, C=final_lr_C, max_iter=final_lr_max_iter,
                                             tol=final_lr_tol, solver='lbfgs', warm_start=True,
                                             random_state=radio_grid_rdstate)

                    clf.fit(radiofeature_train, label_train)
                    # 预测测试集
                    y_prob = clf.predict_proba(radiofeature_test)
                    radioprob_lr = radioprob_lr + list(y_prob[:, 1])
                    important = clf.coef_[0]
                    radio_important.append(list(important))
                    if test_outside:
                        y_prob = clf.predict_proba(radiofeature_out_test)
                        radioprob_lr_out.append(list(y_prob[:, 1]))

                # 保存label --------------------------------------------------------------------------------
                label_CV = label_CV + list(label_test)

            if show_choose == 1:
                prob_lr = dsfrprob_lr
                if test_outside:
                    prob_lr_out = dsfrprob_lr_out
            elif show_choose == 2:
                prob_lr = radioprob_lr
                if test_outside:
                    prob_lr_out = radioprob_lr_out
            elif show_choose == 3:
                prob_lr = (np.array(dsfrprob_lr) + np.array(radioprob_lr)) / 2
                if test_outside:
                    prob_lr_out = (np.array(dsfrprob_lr_out) + np.array(radioprob_lr_out)) / 2

            # 计算&保存内部数据指标
            acc_lr, sen_lr, spc_lr, AUC_lr, bst_lr = getAccSenSpcAuc(label_CV, prob_lr)  #bst = 0.619854

            # 计算外验证指标，注意，计算指标时候使用内部数据的最佳阈值!!!
            if test_outside:
                label_CV_out = list(label_out)
                prob_lr_out = list(np.mean(np.array(prob_lr_out), axis=0))
                acc_lr_out, sen_lr_out, spc_lr_out, AUC_lr_out, bst_lr_out = getAccSenSpcAuc(label_CV_out,prob_lr_out)



            filename = savepath + sep + 'result.txt'
            f = open(filename, 'a+')
            writewords = str(test1) + r'===========================================================' + '\n' + \
                         ', distill_mode : ' + distill_mode + ',' + \
                         '  cvrdstate: ' + str(cvrdstate) +'\n'+ \
                         'radio_grid_rdstate: ' + str(radio_grid_rdstate) + '\n' + \
                         '  dsfr_grid_rdstate: ' + str(dsfr_grid_rdstate) + '\n' + \
                         '  k_fold:' + str(k_fold) + '\n'


            writewords = writewords + r' @ lr   @ AUC: ' + str('%03f' % AUC_lr) + \
                         ',Acc: ' + str('%03f' % acc_lr) + \
                         ',Sen: ' + str('%03f' % sen_lr) + \
                         ',Spc: ' + str('%03f' % spc_lr) + \
                         ',Best_thresold: ' + str('%03f' % bst_lr) + '\n'


            print(writewords)
            f.write(writewords)
            f.close()

            # 保存外验证指标
            if test_outside:
                filename = savepath + sep + 'result.txt'
                f = open(filename, 'a+')
                writewords = '\n' + r'* outside *' + '\n'

                writewords = writewords + r' @ lr   @ outAUC: ' + str('%03f' % AUC_lr_out) + \
                             ',Acc: ' + str('%03f' % acc_lr_out) + \
                             ',Sen: ' + str('%03f' % sen_lr_out) + \
                             ',Spc: ' + str('%03f' % spc_lr_out) + \
                             ',Best_thresold: ' + str('%03f' % bst_lr_out) + '\n'

                print(writewords)
                f.write(writewords)
                f.close()

