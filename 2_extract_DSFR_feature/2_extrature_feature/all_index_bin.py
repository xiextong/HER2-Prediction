from keras import backend as K

seg_thre = 0.5

def precision(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f_t = y_pred_f+K.ones_like(y_pred_f)*(0.5-seg_thre)
    y_pred_f_t = K.round(K.clip(y_pred_f_t, 0, 1))
    true_positives = K.sum(y_true_f * y_pred_f_t)
    predicted_positives = K.sum(K.round(K.clip(y_pred_f_t, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f_t = y_pred_f+K.ones_like(y_pred_f)*(0.5-seg_thre)
    y_pred_f_t = K.round(K.clip(y_pred_f_t, 0, 1))
    true_positives = K.sum(y_true_f * y_pred_f_t)
    possible_positives = K.sum(K.round(K.clip(y_true_f, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fmeasure(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    fmeasure = (2 * p * r) / (p + r)
    return fmeasure


def tp(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f_t = y_pred_f + K.ones_like(y_pred_f) *(0.5-seg_thre)
    y_pred_f_t = K.round(K.clip(y_pred_f_t, 0, 1))
    tp = K.sum(y_true_f*y_pred_f_t)
    return tp

def fn(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f_t = y_pred_f + K.ones_like(y_pred_f) *(0.5-seg_thre)
    y_pred_f_t = K.round(K.clip(y_pred_f_t, 0, 1))
    tp = K.sum(y_true_f*y_pred_f_t)
    y_true_point = K.sum(K.round(K.clip(y_true_f,0,1)))
    fn = y_true_point-tp
    return fn

def fp(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f_t = y_pred_f + K.ones_like(y_pred_f) * (0.5-seg_thre)
    y_pred_f_t = K.round(K.clip(y_pred_f_t, 0, 1))
    tp = K.sum(y_true_f*y_pred_f_t)
    y_pred_point = K.sum(y_pred_f_t)
    fp = y_pred_point-tp
    return fp

def tn(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_mid_1 = K.ones_like(y_true_f)
    all_point = K.sum(y_mid_1)
    tpp = tp(y_true,y_pred)
    fpp = fp(y_true,y_pred)
    fnp = fn(y_true,y_pred)
    tn = all_point - tpp - fpp - fnp
    return tn

def yt_sum(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_true_s = K.sum(y_true_f)
    return y_true_s

def yt(y_true, y_pred):
    return y_true

def yp(y_true,y_pred):
    return y_pred

def DiceCoef(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def gtc(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_gt_point = K.sum(y_true_f)
    y_bg_point = K.sum((K.ones_like(y_true_f)-y_true_f))
    y_pred_gt = y_pred_f*y_true_f+K.ones_like(y_true_f)*0.00001
    y_pred_bg = ((y_pred_f*K.ones_like(y_true_f)-y_true_f))+y_true_f+K.ones_like(y_true_f)*0.00001
    gt_c = (K.sum(K.binary_crossentropy(y_true_f,y_pred_gt)))/(y_gt_point+1)
    return gt_c*1.2

def bgc(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_gt_point = K.sum(y_true_f)
    y_bg_point = K.sum((K.ones_like(y_true_f)-y_true_f))
    y_pred_gt = y_pred_f*y_true_f+K.ones_like(y_true_f)*0.00001
    y_pred_bg = ((y_pred_f*K.ones_like(y_true_f)-y_true_f))+y_true_f+K.ones_like(y_true_f)*0.00001
    bg_c = (K.sum(K.binary_crossentropy(y_true_f,y_pred_bg)))/(y_bg_point+1)
    return bg_c
