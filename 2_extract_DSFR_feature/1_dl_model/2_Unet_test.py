#keras==2.1.6
#tensorflow==1.7.0
import re
import time
from keras.optimizers import *
from keras.backend.tensorflow_backend import set_session
from keras import backend as keras
from keras import backend as k
import  SimpleITK as sitk
from all_loss_bin import *
from all_index_bin import *
from all_model_bin import *
from openpyxl import workbook
import openpyxl
import pickle

# 设置现存占用比例
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

def extract_data_name(path_name):
    pattern = re.compile('.*?/[0-9]+/(.*?).h5', re.S)
    return re.findall(pattern, path_name)[0]

Order = True

Version = '219_patients'
lr = 10e-4
state = "test_out"
###############################################

# the path control center
#數據讀取路径
PathH5 = r'../bc/data_set/C6/219class_0.8_0.8_0.8_small'
#保存结果路径
result_path = os.path.join('../bc/reault/0.8_0.8_0.8_smallimg112_nor_unet/C6_0.001/ori/class_result/26/test_result/',state+'_'+Version)
if os.path.exists(result_path) == 0:
    os.makedirs(result_path)
#读取测试/验证集患者编号
listPatient = []
all_model = []
model_path = '../bc/reault/0.8_0.8_0.8_smallimg112_nor_unet/C6_0.001/ori/model'
img_save_Path_root = '../bc/reault/0.8_0.8_0.8_smallimg112_nor_unet/C6_0.001/ori/class_result/26/test_result/img'
#listPatient = ['14','39','57','60','86','87','96','104']
# listPatient = ['1','2','3','4','5','6','7','8','9','10','12','13','11','15','17','19','20','21','22','24','25','26',
#                '27','29','31','32','33','34','35','36','37','45','47','53','46','56','75','77','78','76','79','80','81']

# listPatient = ['1','3','4','8','9','10','11','13']
# weight_path = "/media/ds/新加卷/XS_experiment/2D_Unet_result/81_patients_Kfold_2/train_model/XS_model_20190114_142.h5"
# path = '18_1'

f = open('../bc/data_set/all_label.pkl','rb')
all_label = pickle.load(f)
seg_list = [pth_id for pth_id in all_label.keys() if all_label[pth_id]['seg']=='Y']

for root, sub_dirs, filelist in os.walk(PathH5):
    for filename in filelist:
        pth_id = int(filename.split('_')[0])
        if pth_id not in seg_list:
            if 'MRI' in filename:
                listPatient.append(os.path.join(root,filename))



for root, sub_dirs, filelist in os.walk(model_path):
    for filename in filelist:
        all_model.append(root + '/' + filename)
all_model.sort(key=lambda x: int(x.split('/')[-1].split('-')[-1].split('.')[0]))

print(listPatient)
vailset_num = len(listPatient)    #所有分割数据h5文件
#np.random.shuffle(listPatient)   #打乱图片顺序

##############################
height = 64
width = 64
batchnum = 1
data_c = np.zeros([batchnum,height,width,1],dtype=float)
label_c = np.zeros([batchnum,height,width,1],dtype =float)
data_c_test = np.zeros([1,height,width,1],dtype=float)
label_c_test = np.zeros([1,height,width,1],dtype=float)
b= np.zeros([height,width],dtype=float)
b1= np.zeros([1,height,width,1],dtype=float)
b2= np.zeros([1,height,width,1],dtype=float)
result_img = np.zeros([1,height,width,1],dtype=float)
result_final = np.zeros([height,width],dtype=float)
save_img = np.zeros([height, width, 1], dtype=float)
################################



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

def gtc_bin(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_gt_point = K.sum(y_true_f)
    y_bg_point = K.sum((K.ones_like(y_true_f)-y_true_f))
    y_pred_gt = y_pred_f*y_true_f+K.ones_like(y_true_f)*0.0000001
    y_pred_bg = (y_true_f-K.ones_like(y_true_f)*0.0000001)+y_pred_f*(K.ones_like(y_true_f)-y_true_f)
    gt_c = (K.sum(K.binary_crossentropy(y_true_f,y_pred_gt)))/(y_gt_point+1)
    return gt_c*0.1

def bgc_bin(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_gt_point = K.sum(y_true_f)
    y_bg_point = K.sum((K.ones_like(y_true_f)-y_true_f))
    y_pred_gt = y_pred_f*y_true_f+K.ones_like(y_true_f)*0.0000001
    y_pred_bg = (y_true_f-K.ones_like(y_true_f)*0.0000001)+y_pred_f*(K.ones_like(y_true_f)-y_true_f)
    bg_c = (K.sum(K.binary_crossentropy(y_true_f,y_pred_bg)))/(y_bg_point+1)
    return bg_c*0.9

def gt_sum(y_true,y_pred):
    gt_sum = K.sum(K.flatten(y_true))
    return gt_sum
def result_calculate(save_img,label_d):
    #print(label_d.shape)
    y = save_img.shape[1]
    x = save_img.shape[0]
    seg_thre = 0.2
    tp, tn, fp, fn, other = 0, 0, 0, 0, 0

    pred = save_img
    roi = label_d

    _, pred1 = cv2.threshold(pred, seg_thre, 1, cv2.THRESH_BINARY)
    # print(x)
    # print(y)
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
def get_accuracy(SR,GT,threshold=0.5):
    # SR = SR.view(-1)
    # GT = GT.view(-1)
    SR = SR.reshape([1,SR.shape[0]*SR.shape[1]*SR.shape[2]])
    GT = GT.reshape([1,GT.shape[0]*GT.shape[1]*GT.shape[2]])
    # SR = SR.numpy()
    # GT = GT.numpy()

    SR = SR > threshold
    GT = GT == np.max(GT)
    corr = np.sum(SR==GT)

    acc = float(corr)/float(SR.shape[0])

    return acc

def get_sensitivity(SR,GT,threshold=0.5):
    SR = SR.reshape([1, SR.shape[0] * SR.shape[1] * SR.shape[2]])
    GT = GT.reshape([1, GT.shape[0] * GT.shape[1] * GT.shape[2]])

    SR = (SR > threshold).astype(np.float)
    GT = (GT == np.max(GT)).astype(np.float)

    # TP : True Positive
    # FN : False Negative
    TP = (((SR == 1.).astype(np.float) + (GT == 1.).astype(np.float)) == 2.).astype(np.float)
    FN = (((SR == 0.).astype(np.float) + (GT == 1.).astype(np.float)) == 2.).astype(np.float)

    SE = float(np.sum(TP)) / (float(np.sum(TP + FN)) + 1e-6)

    return SE

def get_specificity(SR,GT,threshold=0.5):
    SR = SR.reshape([1, SR.shape[0] * SR.shape[1] * SR.shape[2]])
    GT = GT.reshape([1, GT.shape[0] * GT.shape[1] * GT.shape[2]])

    SR = (SR > threshold).astype(np.float)
    GT = (GT == np.max(GT)).astype(np.float)

    # TN : True Negative
    # FP : False Positive
    TN = (((SR == 0.).astype(np.float) + (GT == 0.).astype(np.float)) == 2.).astype(np.float)
    FP = (((SR == 1.).astype(np.float) + (GT == 0.).astype(np.float)) == 2.).astype(np.float)

    SP = float(np.sum(TN)) / (float(np.sum(TN + FP)) + 1e-6)

    return SP

def get_DC(SR,GT,threshold=0.5):

    SR = SR.reshape([1, SR.shape[0] * SR.shape[1] * SR.shape[2]])
    GT = GT.reshape([1, GT.shape[0] * GT.shape[1] * GT.shape[2]])


    SR = (SR > threshold).astype(np.float)
    GT = (GT == np.max(GT)).astype(np.float)

    Inter = np.sum(((SR+GT)==2).astype(np.float))
    DC = float(2*Inter)/(float(np.sum(SR)+np.sum(GT)) + 1e-6)

    return DC

if __name__ == "__main__":

    model = twoDNet((height, width))
    liuzhou = False
    model.compile(optimizer=Adam(lr=lr), loss=focal_loss_fixed2,metrics=['acc', precision, recall, fmeasure, tp, fp, tn, fn, yt_sum])
    data_name = '/' + state
    #  创建Excel表并写入数据
    if os.path.exists(os.path.join(result_path)) == 0:
        os.makedirs(os.path.join(result_path))
    exc_name = os.path.join(result_path,'all_dsc.xlsx')
    ws = []  # 全局工作表对象
    wb = workbook.Workbook()  # 创建Excel对象
    ws = wb.active  # 获取当前正在操作的表对象
    ws.append(['model', 'sen', 'spe', 'acc',' dsc'])
    wb.save(exc_name)
    exc = openpyxl.load_workbook(exc_name)

    # 结果写入文档
    Result_save_Path = result_path
    if os.path.exists(Result_save_Path) == 0:
        os.makedirs(Result_save_Path)
    for i in range(26,27):
        #保存结果
        weight_path = all_model[i]
        print(weight_path)
        model.load_weights(weight_path, by_name=False)

        test_num = 0
        num_count = 0
        batch_count = 0
        acc_all = 0
        dsc_all = 0
        spe_all = 0
        sen_all = 0


        time_start = time.time()
        num = 0
        for m_n in range(len(listPatient)):
            #开始计时----------------------------------------------------------------------------------------------------
            if 'MRI' in listPatient[m_n]:
                mri_sitk = sitk.ReadImage(listPatient[m_n])
                roi_sitk = sitk.ReadImage(listPatient[m_n].split('_MRI')[0]+'_ROI.nii.gz')
                #print(listPatient[m_n])
                mri_arry = sitk.GetArrayFromImage(mri_sitk)
                #归一化
                mri_arry = (mri_arry - np.min(mri_arry)) / (np.max(mri_arry) - np.min(mri_arry))
                roi_arry = sitk.GetArrayFromImage(roi_sitk)
                pth_id = int(listPatient[m_n].split('/')[-1].split('_')[0])
                img_save_Path = os.path.join(img_save_Path_root,str(pth_id))
                if not os.path.exists(img_save_Path):
                    os.makedirs(img_save_Path)
                if liuzhou:
                    roi_arry = 1-roi_arry
                spaceing = mri_sitk.GetSpacing()
                ori      = mri_sitk.GetOrigin()
                tranform = mri_sitk.GetDirection()

                zz,yy,xx = mri_arry.shape
                test_num = test_num+1
                pth_name = int(listPatient[m_n].split('/')[-1].split('_')[0])
                predi_img = np.zeros([zz,yy,xx])

            for zzz in range(zz):

                data = np.array(mri_arry[zzz,:,:], dtype=float)
                label = np.array(roi_arry[zzz,:,:], dtype=float)


                _,label = cv2.threshold(label, 0.5, 1, cv2.THRESH_BINARY)

                data_c[batch_count,:,:,0] = data[:,:]
                model_name =all_model[i].split('/')[-1].split('-')[-1].split('.')[0]
                #result_data = model.test_on_batch([data_c],label_c)
                result_img = model.predict_on_batch([data_c])

                save_img = result_img[0, :, :, 0]
                predi_img[zzz,:,:] = save_img
                num_count = num_count + 1

                _, save_img = cv2.threshold(save_img, 0.5, 1, cv2.THRESH_BINARY)

                img_name = img_save_Path + r'/' + str(pth_id)+'_'+str(zzz) + '.jpg'
                cv2.imwrite(img_name, save_img * 255)

            save_img_itk = sitk.GetImageFromArray(predi_img)
            save_img_itk.SetSpacing(spaceing)
            save_img_itk.SetOrigin(ori)
            save_img_itk.SetDirection(tranform)
            sitk_img_save_path = os.path.join(Result_save_Path, 'nii', model_name,str(pth_name) + '_' + str(zzz) + '.nii.gz')

            if os.path.exists(os.path.join(Result_save_Path, 'nii', model_name)) == 0:
                os.makedirs(os.path.join(Result_save_Path, 'nii', model_name))
            sitk.WriteImage(save_img_itk, sitk_img_save_path)

            pre_sitk = sitk.ReadImage(sitk_img_save_path)
            pre_arry1 = sitk.GetArrayFromImage(pre_sitk)
            _, pre_arry = cv2.threshold(pre_arry1, 0.5, 1, cv2.THRESH_BINARY)

            acc1 = get_accuracy(pre_arry, roi_arry)
            sen1 = get_sensitivity(pre_arry, roi_arry)
            spe1 = get_specificity(pre_arry, roi_arry)
            dsc1 = get_DC(pre_arry, roi_arry)

            acc_all = acc_all + acc1
            dsc_all = dsc_all + dsc1
            spe_all = spe_all + spe1
            sen_all = sen_all + sen1
            if os.path.exists(os.path.join(Result_save_Path,'patience_dsc')) == 0:
                os.makedirs(os.path.join(Result_save_Path,'patience_dsc'))
            f_t = open(os.path.join(Result_save_Path,'patience_dsc',model_name+'.txt'),'a+')
            f_t.write('@patient_' + str(pth_name) + '\n@sen:' + str(sen1) + '  @spe:' + str(spe1) + '  @acc:' + str(acc1)+'\n@dsc:' + str(dsc1) + '\n\n')
            f_t.close()
            print('model:{}  patient:{} acc:{}  dsc: {}'.format(str(model_name),str(pth_name),str(acc1),str(dsc1)))
        # 结束计时--------------------------------------------------------------------------------------------------------
        # time_end = time.time()

        dsc_all = dsc_all/test_num
        acc_all = acc_all/test_num
        spe_all = spe_all/test_num
        sen_all = sen_all/test_num

        print('Total_dsc:  ' , dsc_all)

        f2 = open(Result_save_Path+'/TotalResult_dsc.txt', 'a+')
        f2.write('model_name: ' +all_model[i] + '\n'+'dsc: ' +str(dsc_all)  + 'acc: ' +str(acc_all) + '\n'+ 'sen: ' +str(sen_all) + '\n'+ 'spe: ' +str(spe_all) + '\n\n')
        f2.close()
        ws.append([all_model[i], sen_all, spe_all,acc_all, dsc_all])
        wb.save(exc_name)
