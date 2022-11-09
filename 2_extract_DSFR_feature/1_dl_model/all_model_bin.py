from keras.models import *
from all_layer_bin import *
from keras import initializers
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
import tensorflow as tf

def twoDNet(input_shape=(None, None, 1)):

    input1 = Input(shape=(input_shape[0], input_shape[1], 1), name='input1')

    conv1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(input1)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv1_1_GN')(conv1)
    x = Activation('relu')(x)
    conv1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv1_2_GN')(conv1)
    x = Activation('relu')(x)
    pool1 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool1 shape:", pool1.shape)

    conv2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool1)
    print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv2_1_GN')(conv2)
    x = Activation('relu')(x)
    conv2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv2_2_GN')(conv2)
    x = Activation('relu')(x)
    pool2 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool2 shape:", pool2.shape)

    conv3 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(pool2)
    print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv3_1_GN')(conv3)
    x = Activation('relu')(x)
    conv3 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv3_2_GN')(conv3)
    x = Activation('relu')(x)
    pool3 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool3 shape:", pool3.shape)

    conv4 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(pool3)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv4_1_GN')(conv4)
    x = Activation('relu')(x)
    conv4 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv4_2_GN')(conv4)
    x = Activation('relu')(x)
    # drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(x)

    conv5 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(pool4)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv5_1_GN')(conv5)
    x = Activation('relu')(x)
    conv5 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv5_2_GN')(conv5)
    x = Activation('relu')(x)
    # drop5 = Dropout(0.5)(conv5)

    merge6 = concatenate([conv4, (UpSampling2D(size=(2, 2))(x))], axis=3)
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(up6)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv6_1_GN')(conv6)
    x = Activation('relu')(x)
    conv6 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv6_2_GN')(conv6)
    x = Activation('relu')(x)

    merge7 = concatenate([conv3, (UpSampling2D(size=(2, 2))(x))], axis=3)
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(up7)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv7_1_GN')(conv7)
    x = Activation('relu')(x)
    conv7 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv7_2_GN')(conv7)
    x = Activation('relu')(x)

    merge8 = concatenate([conv2, (UpSampling2D(size=(2, 2))(x))],axis=3)
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(up8)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv8_1_GN')(conv8)
    x = Activation('relu')(x)
    conv8 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv8_2_GN')(conv8)
    x = Activation('relu')(x)

    merge9 = concatenate([conv1, (UpSampling2D(size=(2, 2))(x))], axis=3)
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(up9)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv9_1_GN')(conv9)
    x = Activation('relu')(x)
    conv9 = Conv2D(64, 3,  padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv9_2_GN')(conv9)
    x = Activation('relu')(x)

    conv10 = Conv2D(1, 1,kernel_initializer=initializers.TruncatedNormal(mean=0.2,stddev=0.1),activation='sigmoid')(x)

    print('conv10(outputs): ', conv10.shape)
    model = Model(input=[input1], output=[conv10])

    # model.compile(optimizer=SGD(lr=1.0e-8), loss=EuclideanLoss, metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])

    return model
