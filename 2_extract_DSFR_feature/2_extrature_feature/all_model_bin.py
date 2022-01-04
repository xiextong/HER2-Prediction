from keras.models import *
from all_layer_bin import *
from keras import initializers
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
import tensorflow as tf

def Deeplabv3(input_shape=(None, None, 1)):

    OS = 8
    if OS == 8:
        entry_block3_stride = 1
        middle_block_rate = 2  # ! Not mentioned in paper, but required
        exit_block_rates = (2, 4)
        atrous_rates = (12, 24, 36)
    else:
        entry_block3_stride = 2
        middle_block_rate = 1
        exit_block_rates = (1, 2)
        atrous_rates = (6, 12, 18)

    img_input = Input(shape=(input_shape[0], input_shape[1], 1))

    x = Conv2D(32, (3, 3), strides=(2, 2),
               name='entry_flow_conv1_1', use_bias=False, padding='same')(img_input)
    # x = BatchNormalization(name='entry_flow_conv1_1_BN')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_1_GN')(x)
    x = Activation('relu')(x)

    x = conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
    # x = BatchNormalization(name='entry_flow_conv1_2_BN')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_2_GN')(x)
    x = Activation('relu')(x)

    x = xception_block(x, [128, 128, 128], 'entry_flow_block1',
                       skip_connection_type='conv', stride=2,
                       depth_activation=False)
    x, skip1 = xception_block(x, [256, 256, 256], 'entry_flow_block2',
                              skip_connection_type='conv', stride=2,
                              depth_activation=False, return_skip=True)

    x = xception_block(x, [256, 256, 256], 'entry_flow_block3',
                       skip_connection_type='conv', stride=entry_block3_stride,
                       depth_activation=False)
    for i in range(4):
        x = xception_block(x, [256, 256, 256], 'middle_flow_unit_{}'.format(i + 1),
                           skip_connection_type='sum', stride=1, rate=middle_block_rate,
                           depth_activation=False)

    x = xception_block(x, [256, 512, 512], 'exit_flow_block1',
                       skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
                       depth_activation=False)
    x = xception_block(x, [512, 512, 512], 'exit_flow_block2',
                       skip_connection_type='none', stride=1, rate=exit_block_rates[1],
                       depth_activation=True)
    # end of feature extractor

    # branching for Atrous Spatial Pyramid Pooling
    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    # b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = GroupNormalization(groups=2, axis=1, name='aspp0_BN', epsilon=1e-5)(b0)

    b0 = Activation('relu', name='aspp0_activation')(b0)

    # rate = 6 (12)
    b1 = SepConv_BN(x, 256, 'aspp1',
                    rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
    # rate = 12 (24)
    b2 = SepConv_BN(x, 256, 'aspp2',
                    rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
    # rate = 18 (36)
    b3 = SepConv_BN(x, 256, 'aspp3',
                    rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

    # Image Feature branch
    out_shape = int(np.ceil(input_shape[0] / OS))
    b4 = AveragePooling2D(pool_size=(out_shape, out_shape))(x)
    b4 = Conv2D(256, (1, 1), padding='same',
                use_bias=False, name='image_pooling')(b4)
    # b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)
    b4 = BilinearUpsampling((out_shape, out_shape))(b4)

    # concatenate ASPP branches & project
    x = Concatenate()([b4, b0, b1, b2, b3])
    x = Conv2D(256, (1, 1), padding='same',
               use_bias=False, name='concat_projection')(x)
    # x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation('relu')(x)
    # x = Dropout(0.1)(x)

    # DeepLab v.3+ decoder

    # Feature projection
    # x4 (x2) block
    x = BilinearUpsampling(output_size=(int(np.ceil(input_shape[0] / 4)),
                                        int(np.ceil(input_shape[1] / 4))))(x)
    dec_skip1 = Conv2D(48, (1, 1), padding='same',
                       use_bias=False, name='feature_projection0')(skip1)
    # dec_skip1 = BatchNormalization(
    #     name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    dec_skip1 = Activation('relu')(dec_skip1)
    x = Concatenate()([x, dec_skip1])
    x = SepConv_BN(x, 256, 'decoder_conv0',
                   depth_activation=True, epsilon=1e-5)
    x = SepConv_BN(x, 256, 'decoder_conv1',
                   depth_activation=True, epsilon=1e-5)

    x = Conv2D(1, (1, 1), padding='same', name='layer_last_2')(x)
    x = BilinearUpsampling(output_size=(input_shape[0], input_shape[1]))(x)
    # x = Conv2D(1, (1, 1), padding='same',activation='sigmoid', name=last_layer_name)(x)
    x = Activation('linear')(x)


    model = Model(inputs=img_input, output=x, name='deeplabv3+')
    # model.compile(optimizer=Adam(lr=1e-4), loss=['binary_crossentropy'],
    #               metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])
    return model

def threeDNet(input_shape=(None, None, 1)):

    input1 = Input(shape=(input_shape[0], input_shape[1], 1), name='input1')

    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(input1)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv1_1_GN')(conv1)
    x = Activation('relu')(x)
    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv1_2_GN')(conv1)
    x = Activation('relu')(x)
    pool1 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool1 shape:", pool1.shape)

    conv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(pool1)
    print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv2_1_GN')(conv2)
    x = Activation('relu')(x)
    conv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv2_2_GN')(conv2)
    x = Activation('relu')(x)
    pool2 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool2 shape:", pool2.shape)

    conv3 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool2)
    print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv3_1_GN')(conv3)
    x = Activation('relu')(x)
    conv3 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv3_2_GN')(conv3)
    x = Activation('relu')(x)
    pool3 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool3 shape:", pool3.shape)

    conv4 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(pool3)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv4_1_GN')(conv4)
    x = Activation('relu')(x)
    conv4 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv4_2_GN')(conv4)
    x = Activation('relu')(x)
    # drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(x)

    conv5 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(pool4)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv5_1_GN')(conv5)
    x = Activation('relu')(x)
    conv5 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv5_2_GN')(conv5)
    x = Activation('relu')(x)
    # drop5 = Dropout(0.5)(conv5)

    merge6 = concatenate([conv4, (UpSampling2D(size=(2, 2))(x))], axis=3)
    up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(up6)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv6_1_GN')(conv6)
    x = Activation('relu')(x)
    conv6 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv6_2_GN')(conv6)
    x = Activation('relu')(x)

    merge7 = concatenate([conv3, (UpSampling2D(size=(2, 2))(x))], axis=3)
    up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(up7)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv7_1_GN')(conv7)
    x = Activation('relu')(x)
    conv7 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv7_2_GN')(conv7)
    x = Activation('relu')(x)

    merge8 = concatenate([conv2, (UpSampling2D(size=(2, 2))(x))],axis=3)
    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(up8)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv8_1_GN')(conv8)
    x = Activation('relu')(x)
    conv8 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv8_2_GN')(conv8)
    x = Activation('relu')(x)

    merge9 = concatenate([conv1, (UpSampling2D(size=(2, 2))(x))], axis=3)
    up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(up9)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv9_1_GN')(conv9)
    x = Activation('relu')(x)
    conv9 = Conv2D(32, 3,  padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv9_2_GN')(conv9)
    x = Activation('relu')(x)

    conv10 = Conv2D(1, 1,kernel_initializer=initializers.TruncatedNormal(mean=0.2,stddev=0.1),activation='sigmoid')(x)

    model = Model(input=[input1], output=[conv10])

    # model.compile(optimizer=SGD(lr=1.0e-8), loss=EuclideanLoss, metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])

    return model

def threeDNet_test(input_shape=(None, None, 1)):

    input1 = Input(shape=(input_shape[0], input_shape[1], 1), name='input1')

    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(input1)
    print("conv1 shape:", conv1.shape)
    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(conv1)
    print("conv1 shape:", conv1.shape)


    conv10 = Conv2D(1, 1,kernel_initializer=initializers.TruncatedNormal(mean=0.5,stddev=0.5),activation='sigmoid')(conv1)

    model = Model(input=[input1], output=[conv10])

    # model.compile(optimizer=SGD(lr=1.0e-8), loss=EuclideanLoss, metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])

    return model

def DeeplabV2(input_shape=(None, None, 1)):
    Input_img = Input(shape=(input_shape[0], input_shape[1], 1))
    # Input_pet = Input(shape=(image_x, image_y, 1))
    #
    # concat_pet_ct = merge([Input_ct,Input_pet],mode='concat')

    # Block 1
    h = ZeroPadding2D(padding=(1, 1))(Input_img)
    h = Convolution2D(64, 3, 3, name='conv1_1')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_1_GN')(h)
    x = Activation('relu')(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = Convolution2D(64, 3, 3, name='conv1_2')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_2_GN')(h)
    x = Activation('relu')(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

    # Block 2
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(128, 3, 3, name='conv2_1')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv2_1_GN')(h)
    x = Activation('relu')(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_2')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv2_2_GN')(h)
    x = Activation('relu')(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

    # Block 3
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(256, 3, 3, name='conv3_1')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv3_1_GN')(h)
    x = Activation('relu')(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = Convolution2D(256, 3, 3, name='conv3_2')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv3_2_GN')(h)
    x = Activation('relu')(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = Convolution2D(256, 3, 3, name='conv3_3')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv3_3_GN')(h)
    x = Activation('relu')(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

    # Block 4
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(512, 3, 3, name='conv4_1')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv4_1_GN')(h)
    x = Activation('relu')(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = Convolution2D(512, 3, 3, name='conv4_2')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv4_2_GN')(h)
    x = Activation('relu')(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = Convolution2D(512, 3, 3, name='conv4_3')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv4_3_GN')(h)
    x = Activation('relu')(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(h)

    # Block 5
    h = ZeroPadding2D(padding=(2, 2))(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), name='conv5_1')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv5_1_GN')(h)
    x = Activation('relu')(x)
    h = ZeroPadding2D(padding=(2, 2))(x)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), name='conv5_2')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv5_2_GN')(h)
    x = Activation('relu')(x)
    h = ZeroPadding2D(padding=(2, 2))(x)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), name='conv5_3')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv5_3_GN')(h)
    x = Activation('relu')(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    p5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(h)

    # branching for Atrous Spatial Pyramid Pooling
    # hole = 6
    b1 = ZeroPadding2D(padding=(6, 6))(p5)
    b1 = AtrousConvolution2D(1024, 3, 3, atrous_rate=(6, 6), name='fc6_1')(b1)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc6_1_GN')(b1)
    x = Activation('relu')(x)
    # b1 = Dropout(0.5)(b1)
    b1 = Convolution2D(1024, 1, 1, name='fc7_1')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc7_1_GN')(b1)
    x = Activation('relu')(x)
    # b1 = Dropout(0.5)(b1)
    b1 = Convolution2D(1, 1, 1, name='fc8_voc12_1')(x)
    # x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc8_1_GN')(b1)
    # b1 = Activation('relu')(x)

    # hole = 12
    b2 = ZeroPadding2D(padding=(12, 12))(p5)
    b2 = AtrousConvolution2D(1024, 3, 3, atrous_rate=(12, 12), name='fc6_2')(b2)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc6_2_GN')(b2)
    x = Activation('relu')(x)
    # b2 = Dropout(0.5)(b2)
    b2 = Convolution2D(1024, 1, 1, name='fc7_2')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc7_2_GN')(b2)
    x = Activation('relu')(x)
    # b2 = Dropout(0.5)(b2)
    b2 = Convolution2D(1, 1, 1, name='fc8_voc12_2')(x)
    # x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc8_2_GN')(b2)
    # b2 = Activation('relu')(x)

    # hole = 18
    b3 = ZeroPadding2D(padding=(18, 18))(p5)
    b3 = AtrousConvolution2D(1024, 3, 3, atrous_rate=(18, 18), name='fc6_3')(b3)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc6_3_GN')(b3)
    x = Activation('relu')(x)
    # b3 = Dropout(0.5)(b3)
    b3 = Convolution2D(1024, 1, 1, name='fc7_3')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc7_3_GN')(b3)
    x = Activation('relu')(x)
    # b3 = Dropout(0.5)(b3)
    b3 = Convolution2D(1, 1, 1, name='fc8_voc12_3')(x)
    # x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc8_3_GN')(b3)
    # b3 = Activation('relu')(x)

    # hole = 24
    b4 = ZeroPadding2D(padding=(24, 24))(p5)
    b4 = AtrousConvolution2D(1024, 3, 3, atrous_rate=(24, 24), name='fc6_4')(b4)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc6_4_GN')(b4)
    x = Activation('relu')(x)
    # b4 = Dropout(0.5)(b4)
    b4 = Convolution2D(1024, 1, 1, name='fc7_4')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc7_4_GN')(b4)
    x = Activation('relu')(x)
    # b4 = Dropout(0.5)(b4)
    b4 = Convolution2D(1, 1, 1, name='fc8_voc12_4')(x)
    # x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc8_4_GN')(b4)
    # b4 = Activation('relu')(x)

    s = merge([b1, b2, b3, b4], mode='sum')
    bilinear_layer = BilinearUpsampling(output_size=(input_shape[0], input_shape[1]))(s)
    out = Activation('sigmoid')(bilinear_layer)

    model = Model(input=[Input_img], output=out, name='deeplabV2')

    # model.compile(optimizer=Adam(lr=1.0e-5), loss='binary_crossentropy', metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])

    return model

def DeeplabV2_less_parameters(input_shape=(None, None, 1)):
    Input_img = Input(shape=(input_shape[0], input_shape[1], 1))
    # Input_pet = Input(shape=(image_x, image_y, 1))
    #
    # concat_pet_ct = merge([Input_ct,Input_pet],mode='concat')

    # Block 1
    h = ZeroPadding2D(padding=(1, 1))(Input_img)
    h = Convolution2D(64, 3, 3, name='conv1_1')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_1_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = Convolution2D(64, 3, 3, name='conv1_2')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_2_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

    # Block 2
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(64, 3, 3, name='conv2_1')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv2_1_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = Convolution2D(64, 3, 3, activation='relu', name='conv2_2')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv2_2_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

    # Block 3
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(128, 3, 3, name='conv3_1')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv3_1_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = Convolution2D(128, 3, 3, name='conv3_2')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv3_2_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = Convolution2D(128, 3, 3, name='conv3_3')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv3_3_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

    # Block 4
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(128, 3, 3, name='conv4_1')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv4_1_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = Convolution2D(128, 3, 3, name='conv4_2')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv4_2_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = Convolution2D(128, 3, 3, name='conv4_3')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv4_3_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(h)

    # Block 5
    h = ZeroPadding2D(padding=(2, 2))(h)
    h = AtrousConvolution2D(256, 3, 3, atrous_rate=(2, 2), name='conv5_1')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv5_1_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(2, 2))(x)
    h = AtrousConvolution2D(256, 3, 3, atrous_rate=(2, 2), name='conv5_2')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv5_2_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(2, 2))(x)
    h = AtrousConvolution2D(256, 3, 3, atrous_rate=(2, 2), name='conv5_3')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv5_3_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    p5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(h)

    # branching for Atrous Spatial Pyramid Pooling
    # hole = 6
    b1 = ZeroPadding2D(padding=(6, 6))(p5)
    b1 = AtrousConvolution2D(256, 3, 3, atrous_rate=(6, 6), name='fc6_1')(b1)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc6_1_GN')(b1)
    x = LeakyReLU(alpha=0.1)(x)
    # b1 = Dropout(0.5)(b1)
    b1 = Convolution2D(256, 1, 1, name='fc7_1')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc7_1_GN')(b1)
    x = LeakyReLU(alpha=0.1)(x)
    # b1 = Dropout(0.5)(b1)
    b1 = Convolution2D(1, 1, 1, name='fc8_voc12_1')(x)
    # x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc8_1_GN')(b1)
    # b1 = LeakyReLU(alpha=0.1)(x)

    # hole = 12
    b2 = ZeroPadding2D(padding=(12, 12))(p5)
    b2 = AtrousConvolution2D(256, 3, 3, atrous_rate=(12, 12), name='fc6_2')(b2)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc6_2_GN')(b2)
    x = LeakyReLU(alpha=0.1)(x)
    # b2 = Dropout(0.5)(b2)
    b2 = Convolution2D(256, 1, 1, name='fc7_2')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc7_2_GN')(b2)
    x = LeakyReLU(alpha=0.1)(x)
    # b2 = Dropout(0.5)(b2)
    b2 = Convolution2D(1, 1, 1, name='fc8_voc12_2')(x)
    # x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc8_2_GN')(b2)
    # b2 = LeakyReLU(alpha=0.1)(x)

    # hole = 18
    b3 = ZeroPadding2D(padding=(18, 18))(p5)
    b3 = AtrousConvolution2D(256, 3, 3, atrous_rate=(18, 18), name='fc6_3')(b3)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc6_3_GN')(b3)
    x = LeakyReLU(alpha=0.1)(x)
    # b3 = Dropout(0.5)(b3)
    b3 = Convolution2D(256, 1, 1, name='fc7_3')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc7_3_GN')(b3)
    x = LeakyReLU(alpha=0.1)(x)
    # b3 = Dropout(0.5)(b3)
    b3 = Convolution2D(1, 1, 1, name='fc8_voc12_3')(x)
    # x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc8_3_GN')(b3)
    # b3 = LeakyReLU(alpha=0.1)(x)

    # hole = 24
    b4 = ZeroPadding2D(padding=(24, 24))(p5)
    b4 = AtrousConvolution2D(256, 3, 3, atrous_rate=(24, 24), name='fc6_4')(b4)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc6_4_GN')(b4)
    x = LeakyReLU(alpha=0.1)(x)
    # b4 = Dropout(0.5)(b4)
    b4 = Convolution2D(256, 1, 1, name='fc7_4')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc7_4_GN')(b4)
    x = LeakyReLU(alpha=0.1)(x)
    # b4 = Dropout(0.5)(b4)
    b4 = Convolution2D(1, 1, 1, name='fc8_voc12_4')(x)
    # x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc8_4_GN')(b4)
    # b4 = LeakyReLU(alpha=0.1)(x)

    s = merge([b1, b2, b3, b4], mode='sum')
    bilinear_layer = BilinearUpsampling(output_size=(input_shape[0], input_shape[1]))(s)
    conv_out = Convolution2D(1, 1, 1, name='out')(bilinear_layer)
    # out = Activation('linear')(bilinear_layer)

    model = Model(input=[Input_img], output=conv_out, name='deeplabV2')

    # model.compile(optimizer=Adam(lr=1.0e-5), loss='binary_crossentropy', metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])

    return model

def DeeplabV2_multies(input_shape=(None, None, 1)):
    Input_img = Input(shape=(input_shape[0], input_shape[1], 1))
    Input_img2 = Input(shape=(256, 256, 1))
    # Input_pet = Input(shape=(image_x, image_y, 1))
    #
    # concat_pet_ct = merge([Input_ct,Input_pet],mode='concat')

    # Block 1
    h = ZeroPadding2D(padding=(1, 1))(Input_img)
    h = Convolution2D(64, 3, 3, name='conv1_1')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_1_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = Convolution2D(64, 3, 3, name='conv1_2')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_2_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

    # Block 2
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(64, 3, 3, name='conv2_1')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv2_1_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = Convolution2D(64, 3, 3, name='conv2_2')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv2_2_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

    # Block 3
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(128, 3, 3, name='conv3_1')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv3_1_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = Convolution2D(128, 3, 3, name='conv3_2')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv3_2_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = Convolution2D(128, 3, 3, name='conv3_3')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv3_3_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

    # Block 4
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(128, 3, 3, name='conv4_1')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv4_1_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = Convolution2D(128, 3, 3, name='conv4_2')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv4_2_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = Convolution2D(128, 3, 3, name='conv4_3')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv4_3_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(h)

    # Block 1 2
    h = ZeroPadding2D(padding=(1, 1))(Input_img2)
    h = Convolution2D(64, 3, 3, name='conv1_12')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_1_GN2')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = Convolution2D(64, 3, 3, name='conv1_22')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_2_GN2')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

    # Block 2 2
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(64, 3, 3, name='conv2_12')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv2_1_GN2')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = Convolution2D(64, 3, 3, name='conv2_22')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv2_2_GN2')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

    concat_h = concatenate([h1,h2],axis=-1)

    # Block 5
    h = ZeroPadding2D(padding=(2, 2))(concat_h)
    h = AtrousConvolution2D(256, 3, 3, atrous_rate=(2, 2), name='conv5_1')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv5_1_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(2, 2))(x)
    h = AtrousConvolution2D(256, 3, 3, atrous_rate=(2, 2), name='conv5_2')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv5_2_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(2, 2))(x)
    h = AtrousConvolution2D(256, 3, 3, atrous_rate=(2, 2), name='conv5_3')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv5_3_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    p5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(h)

    # branching for Atrous Spatial Pyramid Pooling
    # hole = 6
    b1 = ZeroPadding2D(padding=(6, 6))(p5)
    b1 = AtrousConvolution2D(256, 3, 3, atrous_rate=(6, 6), name='fc6_1')(b1)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc6_1_GN')(b1)
    x = LeakyReLU(alpha=0.1)(x)
    b1 = Dropout(0.2)(x)
    b1 = Convolution2D(256, 1, 1, name='fc7_1')(b1)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc7_1_GN')(b1)
    x = LeakyReLU(alpha=0.1)(x)
    b1 = Dropout(0.2)(x)
    b1 = Convolution2D(1, 1, 1, name='fc8_voc12_1')(b1)
    # x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc8_1_GN')(b1)
    # b1 = LeakyReLU(alpha=0.1)(x)

    # hole = 12
    b2 = ZeroPadding2D(padding=(12, 12))(p5)
    b2 = AtrousConvolution2D(256, 3, 3, atrous_rate=(12, 12), name='fc6_2')(b2)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc6_2_GN')(b2)
    x = LeakyReLU(alpha=0.1)(x)
    b2 = Dropout(0.2)(x)
    b2 = Convolution2D(256, 1, 1, name='fc7_2')(b2)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc7_2_GN')(b2)
    x = LeakyReLU(alpha=0.1)(x)
    b2 = Dropout(0.2)(x)
    b2 = Convolution2D(1, 1, 1, name='fc8_voc12_2')(b2)
    # x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc8_2_GN')(b2)
    # b2 = LeakyReLU(alpha=0.1)(x)

    # hole = 18
    b3 = ZeroPadding2D(padding=(18, 18))(p5)
    b3 = AtrousConvolution2D(256, 3, 3, atrous_rate=(18, 18), name='fc6_3')(b3)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc6_3_GN')(b3)
    x = LeakyReLU(alpha=0.1)(x)
    b3 = Dropout(0.2)(x)
    b3 = Convolution2D(256, 1, 1, name='fc7_3')(b3)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc7_3_GN')(b3)
    x = LeakyReLU(alpha=0.1)(x)
    b3 = Dropout(0.2)(x)
    b3 = Convolution2D(1, 1, 1, name='fc8_voc12_3')(b3)
    # x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc8_3_GN')(b3)
    # b3 = LeakyReLU(alpha=0.1)(x)

    # hole = 24
    b4 = ZeroPadding2D(padding=(24, 24))(p5)
    b4 = AtrousConvolution2D(256, 3, 3, atrous_rate=(24, 24), name='fc6_4')(b4)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc6_4_GN')(b4)
    x = LeakyReLU(alpha=0.1)(x)
    b4 = Dropout(0.2)(x)
    b4 = Convolution2D(256, 1, 1, name='fc7_4')(b4)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc7_4_GN')(b4)
    x = LeakyReLU(alpha=0.1)(x)
    b4 = Dropout(0.2)(x)
    b4 = Convolution2D(1, 1, 1, name='fc8_voc12_4')(b4)
    # x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc8_4_GN')(b4)
    # b4 = LeakyReLU(alpha=0.1)(x)

    s = merge([b1, b2, b3, b4], mode='sum')
    bilinear_layer = BilinearUpsampling(output_size=(input_shape[0], input_shape[1]))(s)
    conv_out = Convolution2D(1, 1, 1, activation='sigmoid',name='out')(bilinear_layer)
    # out = Activation('linear')(bilinear_layer)

    model = Model(input=[Input_img,Input_img2], output=conv_out, name='deeplabV2')

    # model.compile(optimizer=Adam(lr=1.0e-5), loss='binary_crossentropy', metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])

    return model

def DeeplabV2_multies_3D(input_shape=(None, None, 1)):
    Input_img = Input(shape=(input_shape[0], input_shape[1],4, 1))
    Input_img2 = Input(shape=(256, 256,4, 1))
    # Input_pet = Input(shape=(image_x, image_y, 1))
    #
    # concat_pet_ct = merge([Input_ct,Input_pet],mode='concat')

    # Block 1
    h = ZeroPadding3D(padding=(1, 1))(Input_img)
    h = Convolution3D(64, 3, 3, name='conv1_1')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_1_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding3D(padding=(1, 1))(x)
    h = Convolution3D(64, 3, 3, name='conv1_2')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_2_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

    # Block 2
    h = ZeroPadding3D(padding=(1, 1))(h)
    h = Convolution3D(64, 3, 3, name='conv2_1')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv2_1_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding3D(padding=(1, 1))(x)
    h = Convolution3D(64, 3, 3, name='conv2_2')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv2_2_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

    # Block 3
    h = ZeroPadding3D(padding=(1, 1))(h)
    h = Convolution3D(128, 3, 3, name='conv3_1')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv3_1_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding3D(padding=(1, 1))(x)
    h = Convolution3D(128, 3, 3, name='conv3_2')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv3_2_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding3D(padding=(1, 1))(x)
    h = Convolution3D(128, 3, 3, name='conv3_3')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv3_3_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

    # Block 4
    h = ZeroPadding3D(padding=(1, 1))(h)
    h = Convolution3D(128, 3, 3, name='conv4_1')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv4_1_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding3D(padding=(1, 1))(x)
    h = Convolution3D(128, 3, 3, name='conv4_2')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv4_2_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding3D(padding=(1, 1))(x)
    h = Convolution3D(128, 3, 3, name='conv4_3')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv4_3_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(h)

    # Block 1 2
    h = ZeroPadding3D(padding=(1, 1))(Input_img2)
    h = Convolution3D(64, 3, 3, name='conv1_12')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_1_GN2')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding3D(padding=(1, 1))(x)
    h = Convolution3D(64, 3, 3, name='conv1_22')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_2_GN2')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

    # Block 2 2
    h = ZeroPadding3D(padding=(1, 1))(h)
    h = Convolution3D(64, 3, 3, name='conv2_12')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv2_1_GN2')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding3D(padding=(1, 1))(x)
    h = Convolution3D(64, 3, 3, name='conv2_22')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv2_2_GN2')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    h2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

    concat_h = concatenate([h1,h2],axis=-1)

    # Block 5
    h = ZeroPadding3D(padding=(2, 2))(concat_h)
    h = AtrousConvolution3D(256, 3, 3, atrous_rate=(2, 2), name='conv5_1')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv5_1_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding3D(padding=(2, 2))(x)
    h = AtrousConvolution3D(256, 3, 3, atrous_rate=(2, 2), name='conv5_2')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv5_2_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding3D(padding=(2, 2))(x)
    h = AtrousConvolution3D(256, 3, 3, atrous_rate=(2, 2), name='conv5_3')(h)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv5_3_GN')(h)
    x = LeakyReLU(alpha=0.1)(x)
    h = ZeroPadding2D(padding=(1, 1))(x)
    p5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(h)

    # branching for Atrous Spatial Pyramid Pooling
    # hole = 6
    b1 = ZeroPadding3D(padding=(6, 6))(p5)
    b1 = AtrousConvolution3D(256, 3, 3, atrous_rate=(6, 6), name='fc6_1')(b1)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc6_1_GN')(b1)
    x = LeakyReLU(alpha=0.1)(x)
    b1 = Dropout(0.2)(x)
    b1 = Convolution3D(256, 1, 1, name='fc7_1')(b1)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc7_1_GN')(b1)
    x = LeakyReLU(alpha=0.1)(x)
    b1 = Dropout(0.2)(x)
    b1 = Convolution3D(1, 1, 1, name='fc8_voc12_1')(b1)
    # x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc8_1_GN')(b1)
    # b1 = LeakyReLU(alpha=0.1)(x)

    # hole = 12
    b2 = ZeroPadding3D(padding=(12, 12))(p5)
    b2 = AtrousConvolution3D(256, 3, 3, atrous_rate=(12, 12), name='fc6_2')(b2)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc6_2_GN')(b2)
    x = LeakyReLU(alpha=0.1)(x)
    b2 = Dropout(0.2)(x)
    b2 = Convolution3D(256, 1, 1, name='fc7_2')(b2)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc7_2_GN')(b2)
    x = LeakyReLU(alpha=0.1)(x)
    b2 = Dropout(0.2)(x)
    b2 = Convolution3D(1, 1, 1, name='fc8_voc12_2')(b2)
    # x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc8_2_GN')(b2)
    # b2 = LeakyReLU(alpha=0.1)(x)

    # hole = 18
    b3 = ZeroPadding3D(padding=(18, 18))(p5)
    b3 = AtrousConvolution3D(256, 3, 3, atrous_rate=(18, 18), name='fc6_3')(b3)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc6_3_GN')(b3)
    x = LeakyReLU(alpha=0.1)(x)
    b3 = Dropout(0.2)(x)
    b3 = Convolution3D(256, 1, 1, name='fc7_3')(b3)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc7_3_GN')(b3)
    x = LeakyReLU(alpha=0.1)(x)
    b3 = Dropout(0.2)(x)
    b3 = Convolution3D(1, 1, 1, name='fc8_voc12_3')(b3)
    # x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc8_3_GN')(b3)
    # b3 = LeakyReLU(alpha=0.1)(x)

    # hole = 24
    b4 = ZeroPadding3D(padding=(24, 24))(p5)
    b4 = AtrousConvolution3D(256, 3, 3, atrous_rate=(24, 24), name='fc6_4')(b4)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc6_4_GN')(b4)
    x = LeakyReLU(alpha=0.1)(x)
    b4 = Dropout(0.2)(x)
    b4 = Convolution3D(256, 1, 1, name='fc7_4')(b4)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc7_4_GN')(b4)
    x = LeakyReLU(alpha=0.1)(x)
    b4 = Dropout(0.2)(x)
    b4 = Convolution3D(1, 1, 1, name='fc8_voc12_4')(b4)
    # x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_fc8_4_GN')(b4)
    # b4 = LeakyReLU(alpha=0.1)(x)

    s = merge([b1, b2, b3, b4], mode='sum')
    bilinear_layer = BilinearUpsampling(output_size=(input_shape[0], input_shape[1]))(s)
    conv_out = Convolution3D(1, 1, 1, activation='sigmoid',name='out')(bilinear_layer)
    # out = Activation('linear')(bilinear_layer)

    model = Model(input=[Input_img,Input_img2], output=conv_out, name='deeplabV2')

    # model.compile(optimizer=Adam(lr=1.0e-5), loss='binary_crossentropy', metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])

    return model

def threeDNet_less_parameters(input_shape=(None, None, 1)):


    input1 = Input(shape=(input_shape[0], input_shape[1], 1), name='input1')

    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(input1)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_1_GN')(conv1)
    x = LeakyReLU(alpha=0.1,name='L1')(x)
    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_2_GN')(conv1)
    x = LeakyReLU(alpha=0.1,name='L2')(x)
    pool1 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool1 shape:", pool1.shape)

    conv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(pool1)
    print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv2_1_GN')(conv2)
    x = LeakyReLU(alpha=0.1,name='L3')(x)
    conv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv2_2_GN')(conv2)
    x = LeakyReLU(alpha=0.1,name='L4')(x)
    pool2 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool2 shape:", pool2.shape)

    conv3 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(pool2)
    print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv3_1_GN')(conv3)
    x = LeakyReLU(alpha=0.1,name='L5')(x)
    conv3 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv3_2_GN')(conv3)
    x = LeakyReLU(alpha=0.1,name='L6')(x)
    pool3 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool3 shape:", pool3.shape)

    conv4 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool3)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv4_1_GN')(conv4)
    x = LeakyReLU(alpha=0.1,name='L13')(x)
    conv4 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv4_2_GN')(conv4)
    x = LeakyReLU(alpha=0.1,name='L14')(x)
    # drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(x)

    conv5 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(pool4)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv5_1_GN')(conv5)
    x = LeakyReLU(alpha=0.1,name='L15')(x)
    conv5 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv5_2_GN')(conv5)
    x = LeakyReLU(alpha=0.1,name='L16')(x)
    # drop5 = Dropout(0.5)(conv5)

    merge6 = merge([conv4, (UpSampling2D(size=(2, 2))(x))], mode='concat', concat_axis=3)
    up6 = Conv2D(128, 2, padding='same', kernel_initializer='he_normal')(merge6)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv6_0_GN')(up6)
    x = LeakyReLU(alpha=0.1,name='uL17')(x)
    conv6 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv6_1_GN')(conv6)
    x = LeakyReLU(alpha=0.1,name='L17')(x)
    conv6 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv6_2_GN')(conv6)
    x = LeakyReLU(alpha=0.1,name='L18')(x)

    merge7 = merge([conv3, (UpSampling2D(size=(2, 2))(x))], mode='concat', concat_axis=3)
    up7 = Conv2D(64, 2, padding='same', kernel_initializer='he_normal')(merge7)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv7_0_GN')(up7)
    x = LeakyReLU(alpha=0.1,name='uL19')(x)
    conv7 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv7_1_GN')(conv7)
    x = LeakyReLU(alpha=0.1,name='L19')(x)
    conv7 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv7_2_GN')(conv7)
    x = LeakyReLU(alpha=0.1,name='L20')(x)

    merge8 = merge([conv2, (UpSampling2D(size=(2, 2))(x))], mode='concat', concat_axis=3)
    up8 = Conv2D(64, 2, padding='same', kernel_initializer='he_normal')(merge8)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv8_0_GN')(up8)
    x = LeakyReLU(alpha=0.1,name='uL21')(x)
    conv8 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv8_1_GN')(conv8)
    x = LeakyReLU(alpha=0.1,name='L21')(x)
    conv8 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv8_2_GN')(conv8)
    x = LeakyReLU(alpha=0.1,name='L22')(x)

    merge9 = merge([conv1, (UpSampling2D(size=(2, 2))(x))], mode='concat', concat_axis=3)
    up9 = Conv2D(32, 2, padding='same', kernel_initializer='he_normal')(merge9)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv9_0_GN')(up9)
    x = LeakyReLU(alpha=0.1,name='uL23')(x)
    conv9 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv9_1_GN')(conv9)
    x = LeakyReLU(alpha=0.1,name='L23')(x)
    conv9 = Conv2D(32, 3,  padding='same', kernel_initializer='he_normal')(x)
    # x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv9_2_GN')(conv9)
    # x = LeakyReLU(alpha=0.1,name='L24')(x)

    conv10 = Conv2D(1, 1,kernel_initializer=initializers.TruncatedNormal(mean=0.5,stddev=0.1),activation='sigmoid')(conv9)

    model = Model(input=[input1], output=[conv10])

    # model.compile(optimizer=SGD(lr=1.0e-8), loss=EuclideanLoss, metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])

    return model

def threeDNet_multi_res(input_shape=(None, None, 1),down_sample_block = 3,filter_select = 32):

    input1 = Input(shape=(input_shape[0], input_shape[1], 1), name='input1')
    input2 = Input(shape=(256, 256, 1), name = 'input2')

    # for n in range(1,down_sample_block+1):
    #     x1 = conv_block(x1, 3, [filter_select*n, filter_select*n, filter_select*(n+1)], stage=2, block='a'+str(n), strides=(2, 2))
    #     x1 = identity_block(x1, 3, [filter_select*(n+1), filter_select*(n+1), filter_select*(n+1)], stage=2, block='b'+str(n))
    #     x1 = identity_block(x1, 3, [filter_select*(n+1), filter_select*(n+1), filter_select*(n+1)], stage=2, block='c'+str(n))
    #
    # for n in range(1,down_sample_block):
    #     x2 = conv_block(x2, 3, [filter_select*n, filter_select*n, filter_select*(n+1)], stage=2, block='a_2'+str(n), strides=(2, 2))
    #     x2 = identity_block(x2, 3, [filter_select*(n+1), filter_select*(n+1), filter_select*(n+1)], stage=2, block='b_2'+str(n))
    #     x2 = identity_block(x2, 3, [filter_select*(n+1), filter_select*(n+1), filter_select*(n+1)], stage=2, block='c_2'+str(n))

    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(input1)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_1_GN')(conv1)
    x = LeakyReLU(alpha=0.1, name='L1')(x)
    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_2_GN')(conv1)
    x1 = LeakyReLU(alpha=0.1, name='L2')(x)

    conv1_2 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(input2)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_1_GN2')(conv1_2)
    x = LeakyReLU(alpha=0.1,name='L1_2')(x)
    conv1_2 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_2_GN2')(conv1_2)
    x2 = LeakyReLU(alpha=0.1,name='L2_2')(x)

    conv2 = conv_block(x1, 3, [32, 32, 64], stage=2, block='a', strides=(2, 2))
    x1 = identity_block(conv2, 3, [64, 64, 64], stage=2, block='b')
    x1 = identity_block(x1, 3, [64, 64, 64], stage=2, block='c')

    conv3 = conv_block(x1, 3, [64, 64, 128], stage=3, block='a', strides=(2, 2))
    x1 = identity_block(conv3, 3, [128, 128, 128], stage=3, block='b')
    x1 = identity_block(x1, 3, [128, 128, 128], stage=3, block='c')

    conv4 = conv_block(x1, 3, [128, 128, 256], stage=4, block='a', strides=(2, 2))
    x1 = identity_block(conv4, 3, [256, 256, 256], stage=4, block='b')
    x1 = identity_block(x1, 3, [256, 256, 256], stage=4, block='c')

    conv2_2 = conv_block(x2, 3, [32, 32, 64], stage=2, block='a2', strides=(2, 2))
    x2 = identity_block(conv2_2, 3, [64, 64, 64], stage=2, block='b2')
    x2 = identity_block(x2, 3, [64, 64, 64], stage=2, block='c2')

    conv3_2 = conv_block(x2, 3, [64, 64, 128], stage=3, block='a2', strides=(2, 2))
    x2 = identity_block(conv3_2, 3, [128, 128, 128], stage=3, block='b2')
    x2 = identity_block(x2, 3, [128, 128, 128], stage=3, block='c2')

    concat_p = concatenate([x1,x2],axis=-1)

    x = conv_block(concat_p, 3, [128, 128, 256], stage=5, block='a3', strides=(1, 1))
    x = identity_block(x, 3, [256, 256, 256], stage=5, block='b3')
    x = identity_block(x, 3, [256, 256, 256], stage=5, block='c3')

    merge6 = merge([conv3, (UpSampling2D(size=(2, 2))(x))], mode='concat', concat_axis=-1)
    up6 = Conv2D(128, 2, padding='same', kernel_initializer='he_normal')(merge6)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv6_0_GN')(up6)
    x = LeakyReLU(alpha=0.1,name='uL17')(x)
    conv6 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv6_1_GN')(conv6)
    x = LeakyReLU(alpha=0.1,name='L17')(x)
    conv6 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv6_2_GN')(conv6)
    x = LeakyReLU(alpha=0.1,name='L18')(x)

    merge7 = merge([conv2, (UpSampling2D(size=(2, 2))(x))], mode='concat', concat_axis=-1)
    up7 = Conv2D(64, 2, padding='same', kernel_initializer='he_normal')(merge7)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv7_0_GN')(up7)
    x = LeakyReLU(alpha=0.1,name='uL19')(x)
    conv7 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv7_1_GN')(conv7)
    x = LeakyReLU(alpha=0.1,name='L19')(x)
    conv7 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv7_2_GN')(conv7)
    x = LeakyReLU(alpha=0.1,name='L20')(x)

    merge8 = merge([conv1, (UpSampling2D(size=(2, 2))(x))], mode='concat', concat_axis=-1)
    up8 = Conv2D(32, 2, padding='same', kernel_initializer='he_normal')(merge8)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv8_0_GN')(up8)
    x = LeakyReLU(alpha=0.1,name='uL21')(x)
    conv8 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv8_1_GN')(conv8)
    x = LeakyReLU(alpha=0.1,name='L21')(x)
    conv8 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv8_2_GN')(conv8)
    # x = LeakyReLU(alpha=0.1,name='L22')(x)

    out = Conv2D(1, 1,kernel_initializer='he_normal',activation='sigmoid')(x)

    model = Model(input=[input1,input2], output=[out])

    # model.compile(optimizer=SGD(lr=1.0e-8), loss=EuclideanLoss, metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])

    return model

def threeDNet_multi_res_classification(input_shape=(None, None, 3),down_sample_block = 3,filter_select = 32):

    input1 = Input(shape=(input_shape[0], input_shape[1], 3), name='input1')

    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal',name='conv1_1')(input1)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_1_GN')(conv1)
    x = LeakyReLU(alpha=0.1, name='L1')(x)
    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal',name='conv1_2')(x)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_2_GN')(conv1)
    x1 = LeakyReLU(alpha=0.1, name='L2')(x)

    conv2 = conv_block(x1, 3, [32, 32, 64], stage=2, block='a', strides=(2, 2),)
    x1 = identity_block(conv2, 3, [64, 64, 64], stage=2, block='b')
    x1 = identity_block(x1, 3, [64, 64, 64], stage=2, block='c')

    conv3 = conv_block(x1, 3, [64, 64, 128], stage=3, block='a', strides=(2, 2))
    x1 = identity_block(conv3, 3, [128, 128, 128], stage=3, block='b')
    x1 = identity_block(x1, 3, [128, 128, 128], stage=3, block='c')

    conv4 = conv_block(x1, 3, [128, 128, 256], stage=4, block='a', strides=(2, 2))
    x1 = identity_block(conv4, 3, [256, 256, 256], stage=4, block='b')
    x1 = identity_block(x1, 3, [256, 256, 256], stage=4, block='c')

    x = GlobalAveragePooling2D()(x1)
    fcx = Dense(80,kernel_initializer='he_normal',activation='softmax')(x)
    # out = Activation('softmax')(fcx)
    model = Model(input=[input1], output=[fcx])

    # model.compile(optimizer=SGD(lr=1.0e-8), loss=EuclideanLoss, metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])

    return model

def threeDNet_patch_res(input_shape=(None, None, 1),down_sample_block = 3,filter_select = 32):

    input1 = Input(shape=(input_shape[0], input_shape[1], 1), name='input1')

    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(input1)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_1_GN')(conv1)
    x = LeakyReLU(alpha=0.1, name='L1')(x)
    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_2_GN')(conv1)
    x1 = LeakyReLU(alpha=0.1, name='L2')(x)


    conv2 = conv_block(x1, 3, [32, 32, 64], stage=2, block='a', strides=(2, 2))
    x1 = identity_block(conv2, 3, [64, 64, 64], stage=2, block='b')
    x1 = identity_block(x1, 3, [64, 64, 64], stage=2, block='c')

    conv3 = conv_block(x1, 3, [64, 64, 128], stage=3, block='a', strides=(2, 2))
    x1 = identity_block(conv3, 3, [128, 128, 128], stage=3, block='b')
    x1 = identity_block(x1, 3, [128, 128, 128], stage=3, block='c')

    x = conv_block(x1, 3, [128, 128, 256], stage=5, block='a3', strides=(1, 1))
    x = identity_block(x, 3, [256, 256, 256], stage=5, block='b3')
    x = identity_block(x, 3, [256, 256, 256], stage=5, block='c3')

    merge7 = merge([conv2, (UpSampling2D(size=(2, 2))(x))], mode='concat', concat_axis=3)
    up7 = Conv2D(64, 2, padding='same', kernel_initializer='he_normal')(merge7)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv7_0_GN')(up7)
    x = LeakyReLU(alpha=0.1,name='uL19')(x)
    conv7 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv7_1_GN')(conv7)
    x = LeakyReLU(alpha=0.1,name='L19')(x)
    conv7 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv7_2_GN')(conv7)
    x = LeakyReLU(alpha=0.1,name='L20')(x)

    merge8 = merge([conv1, (UpSampling2D(size=(2, 2))(x))], mode='concat', concat_axis=3)
    up8 = Conv2D(32, 2, padding='same', kernel_initializer='he_normal')(merge8)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv8_0_GN')(up8)
    x = LeakyReLU(alpha=0.1,name='uL21')(x)
    conv8 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv8_1_GN')(conv8)
    x = LeakyReLU(alpha=0.1,name='L21')(x)
    conv8 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv8_2_GN')(conv8)
    # x = LeakyReLU(alpha=0.1,name='L22')(x)

    out = Conv2D(1, 1,kernel_initializer=initializers.TruncatedNormal(mean=0.5,stddev=0.1),activation='sigmoid')(x)

    model = Model(input=[input1], output=[out])

    # model.compile(optimizer=SGD(lr=1.0e-8), loss=EuclideanLoss, metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])

    return model

def threeDNet_multi(input_shape=(None, None, 1)):

    input1 = Input(shape=(input_shape[0], input_shape[1], 1), name='input1')
    input2 = Input(shape=(256, 256, 1), name = 'input2')

    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(input1)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_1_GN')(conv1)
    x = LeakyReLU(alpha=0.1,name='L1')(x)
    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_2_GN')(conv1)
    x = LeakyReLU(alpha=0.1,name='L2')(x)
    pool1 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool1 shape:", pool1.shape)

    conv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(pool1)
    print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv2_1_GN')(conv2)
    x = LeakyReLU(alpha=0.1,name='L3')(x)
    conv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv2_2_GN')(conv2)
    x = LeakyReLU(alpha=0.1,name='L4')(x)
    pool2 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool2 shape:", pool2.shape)

    conv3 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(pool2)
    print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv3_1_GN')(conv3)
    x = LeakyReLU(alpha=0.1,name='L5')(x)
    conv3 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv3_2_GN')(conv3)
    x = LeakyReLU(alpha=0.1,name='L6')(x)
    pool3 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool3 shape:", pool3.shape)


    conv1_2 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(input2)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_1_GN2')(conv1_2)
    x = LeakyReLU(alpha=0.1,name='L7')(x)
    conv1_2 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_2_GN2')(conv1_2)
    x = LeakyReLU(alpha=0.1,name='L8')(x)
    pool1_2 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool1 shape:", pool1.shape)

    conv2_2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(pool1_2)
    print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv2_1_GN2')(conv2_2)
    x = LeakyReLU(alpha=0.1,name='L9')(x)
    conv2_2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv2_2_GN2')(conv2_2)
    x = LeakyReLU(alpha=0.1,name='L10')(x)
    pool2_2 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool2 shape:", pool2.shape)

    conv3_2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool2_2)
    print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv3_1_GN2')(conv3_2)
    x = LeakyReLU(alpha=0.1,name='L11')(x)
    conv3_2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv3_2_GN2')(conv3_2)
    x = LeakyReLU(alpha=0.1,name='L12')(x)
    pool3_2 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool3 shape:", pool3.shape)


    conv4 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool3)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv4_1_GN')(conv4)
    x = LeakyReLU(alpha=0.1,name='L13')(x)
    conv4 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv4_2_GN')(conv4)
    x = LeakyReLU(alpha=0.1,name='L14')(x)
    # drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(x)

    concat_p = concatenate([pool4,pool3_2],axis=-1)

    conv5 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(concat_p)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv5_1_GN')(conv5)
    x = LeakyReLU(alpha=0.1,name='L15')(x)
    conv5 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv5_2_GN')(conv5)
    x = LeakyReLU(alpha=0.1,name='L16')(x)
    # drop5 = Dropout(0.5)(conv5)

    merge6 = merge([conv4, (UpSampling2D(size=(2, 2))(x))], mode='concat', concat_axis=3)
    up6 = Conv2D(128, 2, padding='same', kernel_initializer='he_normal')(merge6)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv6_0_GN')(up6)
    x = LeakyReLU(alpha=0.1,name='uL17')(x)
    conv6 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv6_1_GN')(conv6)
    x = LeakyReLU(alpha=0.1,name='L17')(x)
    conv6 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv6_2_GN')(conv6)
    x = LeakyReLU(alpha=0.1,name='L18')(x)

    merge7 = merge([conv3, (UpSampling2D(size=(2, 2))(x))], mode='concat', concat_axis=3)
    up7 = Conv2D(64, 2, padding='same', kernel_initializer='he_normal')(merge7)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv7_0_GN')(up7)
    x = LeakyReLU(alpha=0.1,name='uL19')(x)
    conv7 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv7_1_GN')(conv7)
    x = LeakyReLU(alpha=0.1,name='L19')(x)
    conv7 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv7_2_GN')(conv7)
    x = LeakyReLU(alpha=0.1,name='L20')(x)

    merge8 = merge([conv2, (UpSampling2D(size=(2, 2))(x))], mode='concat', concat_axis=3)
    up8 = Conv2D(64, 2, padding='same', kernel_initializer='he_normal')(merge8)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv8_0_GN')(up8)
    x = LeakyReLU(alpha=0.1,name='uL21')(x)
    conv8 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv8_1_GN')(conv8)
    x = LeakyReLU(alpha=0.1,name='L21')(x)
    conv8 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv8_2_GN')(conv8)
    x = LeakyReLU(alpha=0.1,name='L22')(x)

    merge9 = merge([conv1, (UpSampling2D(size=(2, 2))(x))], mode='concat', concat_axis=3)
    up9 = Conv2D(32, 2, padding='same', kernel_initializer='he_normal')(merge9)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv9_0_GN')(up9)
    x = LeakyReLU(alpha=0.1,name='uL23')(x)
    conv9 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv9_1_GN')(conv9)
    x = LeakyReLU(alpha=0.1,name='L23')(x)
    conv9 = Conv2D(32, 3,  padding='same', kernel_initializer='he_normal')(x)
    # x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv9_2_GN')(conv9)
    # x = LeakyReLU(alpha=0.1,name='L24')(x)

    conv10 = Conv2D(1, 1,kernel_initializer=initializers.TruncatedNormal(mean=0.5,stddev=0.1),activation='sigmoid')(conv9)

    model = Model(input=[input1,input2], output=[conv10])

    # model.compile(optimizer=SGD(lr=1.0e-8), loss=EuclideanLoss, metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])

    return model

def threeDNet_multi_SE(input_shape=(None, None, 1),gn_axis = 3,GN_able = 1,GN_grou = 2,lk_alph = 0.1):

    input1 = Input(shape=(input_shape[0], input_shape[1], 1), name='input1')
    input2 = Input(shape=(256, 256, 1), name = 'input2')

    c1 = conv_gn_leaky(input1, 3, 32, stage=1, block='a',gn_able = GN_able,gn_group=GN_grou,leaky_alpha = lk_alph, strides=(1, 1))
    c1 = conv_gn_leaky(c1, 3, 32, stage=1, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph, strides=(1, 1))
    pool1 = MaxPooling2D(pool_size=(2, 2))(c1)

    c2 = conv_gn_leaky(pool1, 3, 64, stage=2, block='a',gn_able = GN_able,gn_group=GN_grou,leaky_alpha = lk_alph, strides=(1, 1))
    c2 = conv_gn_leaky(c2, 3, 64, stage=2, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph, strides=(1, 1))
    pool2 = MaxPooling2D(pool_size=(2, 2))(c2)

    c3 = conv_gn_leaky(pool2, 3, 128, stage=3, block='a',gn_able = GN_able,gn_group=GN_grou,leaky_alpha = lk_alph, strides=(1, 1))
    c3 = conv_gn_leaky(c3, 3, 128, stage=3, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph, strides=(1, 1))
    pool3 = MaxPooling2D(pool_size=(2, 2))(c3)

    c1_2 = conv_gn_leaky(input2, 3, 32, stage=1, block='a2',gn_able = GN_able,gn_group=GN_grou,leaky_alpha = lk_alph, strides=(1, 1))
    c1_2 = conv_gn_leaky(c1_2, 3, 32, stage=1, block='b2', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph, strides=(1, 1))
    pool1_2 = MaxPooling2D(pool_size=(2, 2))(c1_2)

    c2_2 = conv_gn_leaky(pool1_2, 3, 64, stage=2, block='a2',gn_able = GN_able,gn_group=GN_grou,leaky_alpha = lk_alph, strides=(1, 1))
    c2_2 = conv_gn_leaky(c2_2, 3, 64, stage=2, block='b2', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph, strides=(1, 1))
    pool2_2 = MaxPooling2D(pool_size=(2, 2))(c2_2)

    c3_2 = conv_gn_leaky(pool2_2, 3, 128, stage=3, block='a2',gn_able = GN_able,gn_group=GN_grou,leaky_alpha = lk_alph, strides=(1, 1))
    c3_2 = conv_gn_leaky(c3_2, 3, 128, stage=3, block='b2', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph, strides=(1, 1))
    pool3_2 = MaxPooling2D(pool_size=(2, 2))(c3_2)

    c4 = conv_gn_leaky(pool3, 3, 256, stage=4, block='a',gn_able = GN_able,gn_group=GN_grou,leaky_alpha = lk_alph, strides=(1, 1))
    c4 = conv_gn_leaky(c4, 3, 256, stage=4, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph, strides=(1, 1))
    pool4 = MaxPooling2D(pool_size=(2, 2))(c4)

    concat_p = concatenate([pool4,pool3_2],axis=-1)

    c5 = conv_gn_leaky(concat_p, 3, 512, stage=5, block='a', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c5 = squeeze_excite_block(c5,ratio=8)
    c5 = conv_gn_leaky(c5, 3, 512, stage=5, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c5 = squeeze_excite_block(c5, ratio=8)



    merge6 = merge([c4, (UpSampling2D(size=(2, 2))(c5))], mode='concat', concat_axis=3)
    up6 = conv_gn_leaky(merge6, 3, 256, stage=6, block='a', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c6 = conv_gn_leaky(up6, 3, 256, stage=6, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c6 = conv_gn_leaky(c6, 3, 128, stage=6, block='c', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))

    merge7 = merge([c3, (UpSampling2D(size=(2, 2))(c6))], mode='concat', concat_axis=3)
    up7 = conv_gn_leaky(merge7, 3, 128, stage=7, block='a', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c7 = conv_gn_leaky(up7, 3, 128, stage=7, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c7 = conv_gn_leaky(c7, 3, 64, stage=7, block='c', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))

    merge8 = merge([c2, (UpSampling2D(size=(2, 2))(c7))], mode='concat', concat_axis=3)
    up8 = conv_gn_leaky(merge8, 3, 64, stage=8, block='a', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c8 = conv_gn_leaky(up8, 3, 64, stage=8, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c8 = conv_gn_leaky(c8, 3, 32, stage=8, block='c', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))

    merge9 = merge([c1, (UpSampling2D(size=(2, 2))(c8))], mode='concat', concat_axis=3)
    up9 = conv_gn_leaky(merge9, 3, 32, stage=9, block='a', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c9 = conv_gn_leaky(up9, 3, 32, stage=9, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c9 = conv_gn_leaky(c9, 3, 32, stage=9, block='c', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))

    conv10 = Conv2D(1, 1,kernel_initializer=initializers.TruncatedNormal(mean=0.5,stddev=0.1),activation='sigmoid')(c9)

    model = Model(input=[input1,input2], output=[conv10])

    # model.compile(optimizer=SGD(lr=1.0e-8), loss=EuclideanLoss, metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])

    return model

def threeDNet_multi_SE_deep_supervise(input_shape=(None, None, 1),gn_axis = 3,GN_able = 1,GN_grou = 2,lk_alph = 0.1):

    input1 = Input(shape=(input_shape[0], input_shape[1], 1), name='input1')
    input2 = Input(shape=(256, 256, 1), name = 'input2')

    c1 = conv_gn_leaky(input1, 3, 32, stage=1, block='a',gn_able = GN_able,gn_group=GN_grou,leaky_alpha = lk_alph, strides=(1, 1))
    c1 = conv_gn_leaky(c1, 3, 32, stage=1, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph, strides=(1, 1))
    pool1 = MaxPooling2D(pool_size=(2, 2))(c1)

    c2 = conv_gn_leaky(pool1, 3, 64, stage=2, block='a',gn_able = GN_able,gn_group=GN_grou,leaky_alpha = lk_alph, strides=(1, 1))
    c2 = conv_gn_leaky(c2, 3, 64, stage=2, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph, strides=(1, 1))
    pool2 = MaxPooling2D(pool_size=(2, 2))(c2)

    c3 = conv_gn_leaky(pool2, 3, 128, stage=3, block='a',gn_able = GN_able,gn_group=GN_grou,leaky_alpha = lk_alph, strides=(1, 1))
    c3 = conv_gn_leaky(c3, 3, 128, stage=3, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph, strides=(1, 1))
    pool3 = MaxPooling2D(pool_size=(2, 2))(c3)

    c1_2 = conv_gn_leaky(input2, 3, 32, stage=1, block='a2',gn_able = GN_able,gn_group=GN_grou,leaky_alpha = lk_alph, strides=(1, 1))
    c1_2 = conv_gn_leaky(c1_2, 3, 32, stage=1, block='b2', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph, strides=(1, 1))
    pool1_2 = MaxPooling2D(pool_size=(2, 2))(c1_2)

    c2_2 = conv_gn_leaky(pool1_2, 3, 64, stage=2, block='a2',gn_able = GN_able,gn_group=GN_grou,leaky_alpha = lk_alph, strides=(1, 1))
    c2_2 = conv_gn_leaky(c2_2, 3, 64, stage=2, block='b2', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph, strides=(1, 1))
    pool2_2 = MaxPooling2D(pool_size=(2, 2))(c2_2)

    c3_2 = conv_gn_leaky(pool2_2, 3, 128, stage=3, block='a2',gn_able = GN_able,gn_group=GN_grou,leaky_alpha = lk_alph, strides=(1, 1))
    c3_2 = conv_gn_leaky(c3_2, 3, 128, stage=3, block='b2', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph, strides=(1, 1))
    # pool3_2 = MaxPooling2D(pool_size=(2, 2))(c3_2)

    c4 = conv_gn_leaky(pool3, 3, 256, stage=4, block='a',gn_able = GN_able,gn_group=GN_grou,leaky_alpha = lk_alph, strides=(1, 1))
    c4 = conv_gn_leaky(c4, 3, 256, stage=4, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph, strides=(1, 1))
    # pool4 = MaxPooling2D(pool_size=(2, 2))(c4)

    concat_p = concatenate([c4,c3_2],axis=-1)
    b4 = ZeroPadding2D(padding=(4, 4))(concat_p)
    b4 = AtrousConvolution2D(256, 3, 3, atrous_rate=(4, 4), name='fc6_4')(b4)
    x = GroupNormalization(groups=2, axis=3, epsilon=0.1, name='entry_flow_fc6_4_GN')(b4)
    x = LeakyReLU(alpha=0.1)(x)
    c5 = conv_gn_leaky(x, 3, 256, stage=5, block='a', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c5 = squeeze_excite_block(c5,ratio=8)
    c5 = conv_gn_leaky(c5, 3, 256, stage=5, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c5 = squeeze_excite_block(c5, ratio=8)
    c5 = conv_gn_leaky(c5, 3, 256, stage=5, block='c', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c5 = squeeze_excite_block(c5, ratio=8)
    c5 = conv_gn_leaky(c5, 3, 256, stage=5, block='d', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c5 = squeeze_excite_block(c5, ratio=8)


    merge6 = merge([c4, (UpSampling2D(size=(1, 1))(c5))], mode='concat', concat_axis=3)
    up6 = conv_gn_leaky(merge6, 3, 256, stage=6, block='a', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c6 = conv_gn_leaky(up6, 3, 256, stage=6, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c6 = conv_gn_leaky(c6, 3, 128, stage=6, block='c', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))

    merge7 = merge([c3, (UpSampling2D(size=(2, 2))(c6))], mode='concat', concat_axis=3)
    up7 = conv_gn_leaky(merge7, 3, 128, stage=7, block='a', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c7 = conv_gn_leaky(up7, 3, 128, stage=7, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c7 = conv_gn_leaky(c7, 3, 64, stage=7, block='c', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))

    merge8 = merge([c2, (UpSampling2D(size=(2, 2))(c7))], mode='concat', concat_axis=3)
    up8 = conv_gn_leaky(merge8, 3, 64, stage=8, block='a', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c8 = conv_gn_leaky(up8, 3, 64, stage=8, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c8 = conv_gn_leaky(c8, 3, 32, stage=8, block='c', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))

    merge9 = merge([c1, (UpSampling2D(size=(2, 2))(c8))], mode='concat', concat_axis=3)
    up9 = conv_gn_leaky(merge9, 3, 32, stage=9, block='a', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c9 = conv_gn_leaky(up9, 3, 32, stage=9, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c9 = conv_gn_leaky(c9, 3, 32, stage=9, block='c', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=0.5,
                      strides=(1, 1))

    c5_deep =  UpSampling2D(size=(8, 8))(c5)
    c5_deep = Conv2D(1, 1, kernel_initializer=initializers.TruncatedNormal(mean=0.5, stddev=0.1), activation='sigmoid')(
        c5_deep)

    conv10 = Conv2D(1, 1,kernel_initializer=initializers.TruncatedNormal(mean=0.5,stddev=0.1),activation='sigmoid')(c9)

    model = Model(input=[input1,input2], output=[conv10,c5_deep])

    # model.compile(optimizer=SGD(lr=1.0e-8), loss=EuclideanLoss, metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])

    return model

def threeDNet_multi_SE_deep_supervise_test(input_shape=(None, None, 1),gn_axis = 3,GN_able = 1,GN_grou = 2,lk_alph = 0.1):

    input1 = Input(shape=(input_shape[0], input_shape[1], 1), name='input1')
    input2 = Input(shape=(256, 256, 1), name = 'input2')

    c1 = conv_gn_leaky(input1, 3, 32, stage=1, block='a',gn_able = GN_able,gn_group=GN_grou,leaky_alpha = lk_alph, strides=(1, 1))
    c1 = conv_gn_leaky(c1, 3, 32, stage=1, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph, strides=(1, 1))
    pool1 = MaxPooling2D(pool_size=(2, 2))(c1)

    c2 = conv_gn_leaky(pool1, 3, 64, stage=2, block='a',gn_able = GN_able,gn_group=GN_grou,leaky_alpha = lk_alph, strides=(1, 1))
    c2 = conv_gn_leaky(c2, 3, 64, stage=2, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph, strides=(1, 1))
    pool2 = MaxPooling2D(pool_size=(2, 2))(c2)

    c3 = conv_gn_leaky(pool2, 3, 128, stage=3, block='a',gn_able = GN_able,gn_group=GN_grou,leaky_alpha = lk_alph, strides=(1, 1))
    c3 = conv_gn_leaky(c3, 3, 128, stage=3, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph, strides=(1, 1))
    pool3 = MaxPooling2D(pool_size=(2, 2))(c3)

    c1_2 = conv_gn_leaky(input2, 3, 32, stage=1, block='a2',gn_able = GN_able,gn_group=GN_grou,leaky_alpha = lk_alph, strides=(1, 1))
    c1_2 = conv_gn_leaky(c1_2, 3, 32, stage=1, block='b2', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph, strides=(1, 1))
    pool1_2 = MaxPooling2D(pool_size=(2, 2))(c1_2)

    c2_2 = conv_gn_leaky(pool1_2, 3, 64, stage=2, block='a2',gn_able = GN_able,gn_group=GN_grou,leaky_alpha = lk_alph, strides=(1, 1))
    c2_2 = conv_gn_leaky(c2_2, 3, 64, stage=2, block='b2', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph, strides=(1, 1))
    pool2_2 = MaxPooling2D(pool_size=(2, 2))(c2_2)

    c3_2 = conv_gn_leaky(pool2_2, 3, 128, stage=3, block='a2',gn_able = GN_able,gn_group=GN_grou,leaky_alpha = lk_alph, strides=(1, 1))
    c3_2 = conv_gn_leaky(c3_2, 3, 128, stage=3, block='b2', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph, strides=(1, 1))
    # pool3_2 = MaxPooling2D(pool_size=(2, 2))(c3_2)

    c4 = conv_gn_leaky(pool3, 3, 256, stage=4, block='a',gn_able = GN_able,gn_group=GN_grou,leaky_alpha = lk_alph, strides=(1, 1))
    c4 = conv_gn_leaky(c4, 3, 256, stage=4, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph, strides=(1, 1))
    # pool4 = MaxPooling2D(pool_size=(2, 2))(c4)

    concat_p = concatenate([c4,c3_2],axis=-1)
    b4 = ZeroPadding2D(padding=(4, 4))(concat_p)
    b4 = AtrousConvolution2D(256, 3, 3, atrous_rate=(4, 4), name='fc6_4')(b4)
    x = GroupNormalization(groups=2, axis=3, epsilon=0.1, name='entry_flow_fc6_4_GN')(b4)
    x = LeakyReLU(alpha=0.1)(x)
    c5 = conv_gn_leaky(x, 3, 256, stage=5, block='a', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c5 = squeeze_excite_block(c5,ratio=8)
    c5 = conv_gn_leaky(c5, 3, 256, stage=5, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c5 = squeeze_excite_block(c5, ratio=8)
    c5 = conv_gn_leaky(c5, 3, 256, stage=5, block='c', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c5 = squeeze_excite_block(c5, ratio=8)
    c5 = conv_gn_leaky(c5, 3, 256, stage=5, block='d', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c5 = squeeze_excite_block(c5, ratio=8)


    merge6 = merge([c4, (UpSampling2D(size=(1, 1))(c5))], mode='concat', concat_axis=3)
    up6 = conv_gn_leaky(merge6, 3, 256, stage=6, block='a', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c6 = conv_gn_leaky(up6, 3, 256, stage=6, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c6 = conv_gn_leaky(c6, 3, 128, stage=6, block='c', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))

    merge7 = merge([c3, (UpSampling2D(size=(2, 2))(c6))], mode='concat', concat_axis=3)
    up7 = conv_gn_leaky(merge7, 3, 128, stage=7, block='a', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c7 = conv_gn_leaky(up7, 3, 128, stage=7, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c7 = conv_gn_leaky(c7, 3, 64, stage=7, block='c', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))

    merge8 = merge([c2, (UpSampling2D(size=(2, 2))(c7))], mode='concat', concat_axis=3)
    up8 = conv_gn_leaky(merge8, 3, 64, stage=8, block='a', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c8 = conv_gn_leaky(up8, 3, 64, stage=8, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c8 = conv_gn_leaky(c8, 3, 32, stage=8, block='c', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))

    merge9 = merge([c1, (UpSampling2D(size=(2, 2))(c8))], mode='concat', concat_axis=3)
    up9 = conv_gn_leaky(merge9, 3, 32, stage=9, block='a', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c9 = conv_gn_leaky(up9, 3, 32, stage=9, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c9 = conv_gn_leaky(c9, 3, 32, stage=9, block='c', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=0.5,
                      strides=(1, 1))

    c5_deep =  UpSampling2D(size=(8, 8))(c5)
    c5_deep = Conv2D(1, 1, kernel_initializer=initializers.TruncatedNormal(mean=0.5, stddev=0.1), activation='sigmoid')(
        c5_deep)

    conv10 = Conv2D(1, 1,kernel_initializer=initializers.TruncatedNormal(mean=0.5,stddev=0.1),activation='sigmoid')(c9)

    model = Model(input=[input1,input2], output=[conv10])

    # model.compile(optimizer=SGD(lr=1.0e-8), loss=EuclideanLoss, metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])

    return model

def threeDNet_multi_less_pool_SE(input_shape=(None, None, 1)):

    input1 = Input(shape=(input_shape[0], input_shape[1], 1), name='input1')
    input2 = Input(shape=(256, 256, 1), name = 'input2')

    gn_axis = 3

    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(input1)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=gn_axis, epsilon=0.1, name='entry_flow_conv1_1_GN')(conv1)
    x = LeakyReLU(alpha=0.1,name='L1')(x)
    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=gn_axis, epsilon=0.1, name='entry_flow_conv1_2_GN')(conv1)
    x = LeakyReLU(alpha=0.1,name='L2')(x)
    pool1 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool1 shape:", pool1.shape)

    conv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(pool1)
    print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=2, axis=gn_axis, epsilon=0.1, name='entry_flow_conv2_1_GN')(conv2)
    x = LeakyReLU(alpha=0.1,name='L3')(x)
    conv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=2, axis=gn_axis, epsilon=0.1, name='entry_flow_conv2_2_GN')(conv2)
    x = LeakyReLU(alpha=0.1,name='L4')(x)
    pool2 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool2 shape:", pool2.shape)

    conv3 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(pool2)
    print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=2, axis=gn_axis, epsilon=0.1, name='entry_flow_conv3_1_GN')(conv3)
    x = LeakyReLU(alpha=0.1,name='L5')(x)
    conv3 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=2, axis=gn_axis, epsilon=0.1, name='entry_flow_conv3_2_GN')(conv3)
    x = LeakyReLU(alpha=0.1,name='L6')(x)
    pool3 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool3 shape:", pool3.shape)


    conv1_2 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(input2)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=gn_axis, epsilon=0.1, name='entry_flow_conv1_1_GN2')(conv1_2)
    x = LeakyReLU(alpha=0.1,name='L7')(x)
    conv1_2 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=gn_axis, epsilon=0.1, name='entry_flow_conv1_2_GN2')(conv1_2)
    x = LeakyReLU(alpha=0.1,name='L8')(x)
    pool1_2 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool1 shape:", pool1.shape)

    conv2_2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(pool1_2)
    print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=2, axis=gn_axis, epsilon=0.1, name='entry_flow_conv2_1_GN2')(conv2_2)
    x = LeakyReLU(alpha=0.1,name='L9')(x)
    conv2_2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=2, axis=gn_axis, epsilon=0.1, name='entry_flow_conv2_2_GN2')(conv2_2)
    x = LeakyReLU(alpha=0.1,name='L10')(x)
    pool2_2 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool2 shape:", pool2.shape)

    conv3_2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool2_2)
    print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=2, axis=gn_axis, epsilon=0.1, name='entry_flow_conv3_1_GN2')(conv3_2)
    x = LeakyReLU(alpha=0.1,name='L11')(x)
    conv3_2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=2, axis=gn_axis, epsilon=0.1, name='entry_flow_conv3_2_GN2')(conv3_2)
    x3 = LeakyReLU(alpha=0.1,name='L12')(x)
    # pool3_2 = MaxPooling2D(pool_size=(2, 2))(x)
    # print("pool3 shape:", pool3.shape)


    conv4 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool3)
    x = GroupNormalization(groups=2, axis=gn_axis, epsilon=0.1, name='entry_flow_conv4_1_GN')(conv4)
    x = LeakyReLU(alpha=0.1,name='L13')(x)
    conv4 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=gn_axis, epsilon=0.1, name='entry_flow_conv4_2_GN')(conv4)
    x = LeakyReLU(alpha=0.1,name='L14')(x)
    # drop4 = Dropout(0.5)(conv4)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(x)

    concat_p = concatenate([x,x3],axis=-1)

    conv5_1 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(concat_p)
    conv5_1 = GroupNormalization(groups=2, axis=gn_axis, epsilon=0.1, name='entry_flow_conv5_1_GN')(conv5_1)
    conv5_1 = LeakyReLU(alpha=0.1,name='L15')(conv5_1)
    conv5_1_se = GlobalAveragePooling2D()(conv5_1)
    conv5_1_se = Reshape([1,1,256])(conv5_1_se)
    fc5_se = Dense(128,activation='relu',kernel_initializer='he_normal')(conv5_1_se)
    fc5_se = Dense(256,activation='sigmoid',kernel_initializer='he_normal')(fc5_se)
    conv5_1_se_o = multiply([conv5_1,fc5_se])
    conv5_2 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv5_1_se_o)
    conv5_2 = GroupNormalization(groups=2, axis=gn_axis, epsilon=0.1, name='entry_flow_conv5_2_GN')(conv5_2)
    conv5_2 = LeakyReLU(alpha=0.1,name='L16')(conv5_2)
    conv5_2_se = GlobalAveragePooling2D()(conv5_2)
    conv5_2_se = Reshape([1, 1, 256])(conv5_2_se)
    fc5_se = Dense(128,activation='relu',kernel_initializer='he_normal')(conv5_2_se)
    fc5_se = Dense(256,activation='sigmoid',kernel_initializer='he_normal')(fc5_se)
    conv5_2_se_o = multiply([conv5_2,fc5_se])




    # merge6 = merge([conv4, (UpSampling2D(size=(2, 2))(conv5_2_se_o))], mode='concat', concat_axis=3)
    up6 = Conv2D(128, 2, padding='same', kernel_initializer='he_normal')(conv5_2_se_o)
    x = GroupNormalization(groups=2, axis=gn_axis, epsilon=0.1, name='entry_flow_conv6_0_GN')(up6)
    x = LeakyReLU(alpha=0.1,name='uL17')(x)
    conv6 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=gn_axis, epsilon=0.1, name='entry_flow_conv6_1_GN')(conv6)
    x = LeakyReLU(alpha=0.1,name='L17')(x)
    conv6 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=gn_axis, epsilon=0.1, name='entry_flow_conv6_2_GN')(conv6)
    x = LeakyReLU(alpha=0.1,name='L18')(x)

    merge7 = merge([conv3, (UpSampling2D(size=(2, 2))(x))], mode='concat', concat_axis=3)
    up7 = Conv2D(64, 2, padding='same', kernel_initializer='he_normal')(merge7)
    x = GroupNormalization(groups=2, axis=gn_axis, epsilon=0.1, name='entry_flow_conv7_0_GN')(up7)
    x = LeakyReLU(alpha=0.1,name='uL19')(x)
    conv7 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=gn_axis, epsilon=0.1, name='entry_flow_conv7_1_GN')(conv7)
    x = LeakyReLU(alpha=0.1,name='L19')(x)
    conv7 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=gn_axis, epsilon=0.1, name='entry_flow_conv7_2_GN')(conv7)
    x = LeakyReLU(alpha=0.1,name='L20')(x)

    merge8 = merge([conv2, (UpSampling2D(size=(2, 2))(x))], mode='concat', concat_axis=3)
    up8 = Conv2D(64, 2, padding='same', kernel_initializer='he_normal')(merge8)
    x = GroupNormalization(groups=2, axis=gn_axis, epsilon=0.1, name='entry_flow_conv8_0_GN')(up8)
    x = LeakyReLU(alpha=0.1,name='uL21')(x)
    conv8 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=gn_axis, epsilon=0.1, name='entry_flow_conv8_1_GN')(conv8)
    x = LeakyReLU(alpha=0.1,name='L21')(x)
    conv8 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=gn_axis, epsilon=0.1, name='entry_flow_conv8_2_GN')(conv8)
    x = LeakyReLU(alpha=0.1,name='L22')(x)

    merge9 = merge([conv1, (UpSampling2D(size=(2, 2))(x))], mode='concat', concat_axis=3)
    up9 = Conv2D(32, 2, padding='same', kernel_initializer='he_normal')(merge9)
    x = GroupNormalization(groups=2, axis=gn_axis, epsilon=0.1, name='entry_flow_conv9_0_GN')(up9)
    x = LeakyReLU(alpha=0.1,name='uL23')(x)
    conv9 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=gn_axis, epsilon=0.1, name='entry_flow_conv9_1_GN')(conv9)
    x = LeakyReLU(alpha=0.1,name='L23')(x)
    conv9 = Conv2D(32, 3,  padding='same', kernel_initializer='he_normal')(x)
    # x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv9_2_GN')(conv9)
    # x = LeakyReLU(alpha=0.1,name='L24')(x)

    conv10 = Conv2D(1, 1,kernel_initializer=initializers.TruncatedNormal(mean=0.5,stddev=0.1),activation='sigmoid')(conv9)

    model = Model(input=[input1,input2], output=[conv10])

    # model.compile(optimizer=SGD(lr=1.0e-8), loss=EuclideanLoss, metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])

    return model

def threeDNet_multi_SE_res(input_shape=(None, None, 1)):

    input1 = Input(shape=(input_shape[0], input_shape[1], 1), name='input1')
    input2 = Input(shape=(256, 256, 1), name = 'input2')

    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(input1)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=3, epsilon=0.1, name='entry_flow_conv1_1_GN')(conv1)
    x = LeakyReLU(alpha=0.1,name='L1')(x)
    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=3, epsilon=0.1, name='entry_flow_conv1_2_GN')(conv1)
    x = LeakyReLU(alpha=0.1,name='L2')(x)
    pool1 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool1 shape:", pool1.shape)

    conv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(pool1)
    print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=2, axis=3, epsilon=0.1, name='entry_flow_conv2_1_GN')(conv2)
    x = LeakyReLU(alpha=0.1,name='L3')(x)
    conv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=2, axis=3, epsilon=0.1, name='entry_flow_conv2_2_GN')(conv2)
    x = LeakyReLU(alpha=0.1,name='L4')(x)
    pool2 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool2 shape:", pool2.shape)

    conv3 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(pool2)
    print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=2, axis=3, epsilon=0.1, name='entry_flow_conv3_1_GN')(conv3)
    x = LeakyReLU(alpha=0.1,name='L5')(x)
    conv3 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=2, axis=3, epsilon=0.1, name='entry_flow_conv3_2_GN')(conv3)
    x = LeakyReLU(alpha=0.1,name='L6')(x)
    pool3 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool3 shape:", pool3.shape)


    conv1_2 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(input2)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=3, epsilon=0.1, name='entry_flow_conv1_1_GN2')(conv1_2)
    x = LeakyReLU(alpha=0.1,name='L7')(x)
    conv1_2 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=3, epsilon=0.1, name='entry_flow_conv1_2_GN2')(conv1_2)
    x = LeakyReLU(alpha=0.1,name='L8')(x)
    pool1_2 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool1 shape:", pool1.shape)

    conv2_2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(pool1_2)
    print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=2, axis=3, epsilon=0.1, name='entry_flow_conv2_1_GN2')(conv2_2)
    x = LeakyReLU(alpha=0.1,name='L9')(x)
    conv2_2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=2, axis=3, epsilon=0.1, name='entry_flow_conv2_2_GN2')(conv2_2)
    x = LeakyReLU(alpha=0.1,name='L10')(x)
    pool2_2 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool2 shape:", pool2.shape)

    conv3_2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool2_2)
    print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=2, axis=3, epsilon=0.1, name='entry_flow_conv3_1_GN2')(conv3_2)
    x = LeakyReLU(alpha=0.1,name='L11')(x)
    conv3_2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=2, axis=3, epsilon=0.1, name='entry_flow_conv3_2_GN2')(conv3_2)
    input2_final = LeakyReLU(alpha=0.1,name='L12')(x)
    # pool3_2 = MaxPooling2D(pool_size=(2, 2))(x)
    # print("pool3 shape:", pool3.shape)


    conv4 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool3)
    x = GroupNormalization(groups=2, axis=3, epsilon=0.1, name='entry_flow_conv4_1_GN')(conv4)
    x = LeakyReLU(alpha=0.1,name='L13')(x)
    conv4 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=3, epsilon=0.1, name='entry_flow_conv4_2_GN')(conv4)
    input1_final = LeakyReLU(alpha=0.1,name='L14')(x)
    # drop4 = Dropout(0.5)(conv4)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(x)

    concat_p = concatenate([input1_final,input2_final],axis=-1)

    conv5_1 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(concat_p)
    conv5_1 = GroupNormalization(groups=2, axis=3, epsilon=0.1, name='entry_flow_conv5_1_GN')(conv5_1)
    conv5_1 = LeakyReLU(alpha=0.1,name='L15')(conv5_1)
    conv5_1 = identity_block(conv5_1, 3, [128, 128, 128], stage=1, block='a')
    # conv5_1 = identity_block(conv5_1, 3, [256, 256, 256], stage=1, block='b')
    conv5_1_se = squeeze_excite_block(conv5_1)

    # conv5_2 = identity_block(conv5_1_se, 3, [256, 256, 256], stage=2, block='a')
    # conv5_2 = identity_block(conv5_2, 3, [256, 256, 256], stage=2, block='b')
    conv5_2 = identity_block(conv5_1_se, 3, [128, 128, 128], stage=2, block='c')
    conv5_2_se = squeeze_excite_block(conv5_2)

    merge7 = merge([conv3, (UpSampling2D(size=(2, 2))(conv5_2_se))], mode='concat', concat_axis=3)
    up7 = Conv2D(64, 2, padding='same', kernel_initializer='he_normal')(merge7)
    x = GroupNormalization(groups=2, axis=3, epsilon=0.1, name='entry_flow_conv7_0_GN')(up7)
    x = LeakyReLU(alpha=0.1,name='uL19')(x)
    conv7 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=3, epsilon=0.1, name='entry_flow_conv7_1_GN')(conv7)
    x = LeakyReLU(alpha=0.1,name='L19')(x)
    conv7 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=3, epsilon=0.1, name='entry_flow_conv7_2_GN')(conv7)
    x = LeakyReLU(alpha=0.1,name='L20')(x)

    merge8 = merge([conv2, (UpSampling2D(size=(2, 2))(x))], mode='concat', concat_axis=3)
    up8 = Conv2D(64, 2, padding='same', kernel_initializer='he_normal')(merge8)
    x = GroupNormalization(groups=2, axis=3, epsilon=0.1, name='entry_flow_conv8_0_GN')(up8)
    x = LeakyReLU(alpha=0.1,name='uL21')(x)
    conv8 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=3, epsilon=0.1, name='entry_flow_conv8_1_GN')(conv8)
    x = LeakyReLU(alpha=0.1,name='L21')(x)
    conv8 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=3, epsilon=0.1, name='entry_flow_conv8_2_GN')(conv8)
    x = LeakyReLU(alpha=0.1,name='L22')(x)

    merge9 = merge([conv1, (UpSampling2D(size=(2, 2))(x))], mode='concat', concat_axis=3)
    up9 = Conv2D(32, 2, padding='same', kernel_initializer='he_normal')(merge9)
    x = GroupNormalization(groups=2, axis=3, epsilon=0.1, name='entry_flow_conv9_0_GN')(up9)
    x = LeakyReLU(alpha=0.1,name='uL23')(x)
    conv9 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=3, epsilon=0.1, name='entry_flow_conv9_1_GN')(conv9)
    x = LeakyReLU(alpha=0.1,name='L23')(x)
    conv9 = Conv2D(32, 3,  padding='same', kernel_initializer='he_normal')(x)
    # x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv9_2_GN')(conv9)
    # x = LeakyReLU(alpha=0.1,name='L24')(x)

    conv10 = Conv2D(1, 1,kernel_initializer=initializers.TruncatedNormal(mean=0.5,stddev=0.3),activation='sigmoid')(conv9)

    model = Model(input=[input1,input2], output=[conv10])

    # model.compile(optimizer=SGD(lr=1.0e-8), loss=EuclideanLoss, metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])

    return model

def threeDNet_multi_SE_predict_feature(input_shape=(512, 512, 1)):

    input1 = Input(shape=(input_shape[0], input_shape[1], 1), name='input1')
    input2 = Input(shape=(256, 256, 1), name = 'input2')

    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(input1)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_1_GN')(conv1)
    x = LeakyReLU(alpha=0.1,name='L1')(x)
    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_2_GN')(conv1)
    x = LeakyReLU(alpha=0.1,name='L2')(x)
    pool1 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool1 shape:", pool1.shape)

    conv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(pool1)
    print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv2_1_GN')(conv2)
    x = LeakyReLU(alpha=0.1,name='L3')(x)
    conv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv2_2_GN')(conv2)
    x = LeakyReLU(alpha=0.1,name='L4')(x)
    pool2 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool2 shape:", pool2.shape)

    conv3 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(pool2)
    print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv3_1_GN')(conv3)
    x = LeakyReLU(alpha=0.1,name='L5')(x)
    conv3 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv3_2_GN')(conv3)
    x = LeakyReLU(alpha=0.1,name='L6')(x)
    pool3 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool3 shape:", pool3.shape)


    conv1_2 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(input2)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_1_GN2')(conv1_2)
    x = LeakyReLU(alpha=0.1,name='L7')(x)
    conv1_2 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_2_GN2')(conv1_2)
    x = LeakyReLU(alpha=0.1,name='L8')(x)
    pool1_2 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool1 shape:", pool1.shape)

    conv2_2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(pool1_2)
    print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv2_1_GN2')(conv2_2)
    x = LeakyReLU(alpha=0.1,name='L9')(x)
    conv2_2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv2_2_GN2')(conv2_2)
    x = LeakyReLU(alpha=0.1,name='L10')(x)
    pool2_2 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool2 shape:", pool2.shape)

    conv3_2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool2_2)
    print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv3_1_GN2')(conv3_2)
    x = LeakyReLU(alpha=0.1,name='L11')(x)
    conv3_2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv3_2_GN2')(conv3_2)
    x = LeakyReLU(alpha=0.1,name='L12')(x)
    pool3_2 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool3 shape:", pool3.shape)


    conv4 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool3)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv4_1_GN')(conv4)
    x = LeakyReLU(alpha=0.1,name='L13')(x)
    conv4 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv4_2_GN')(conv4)
    x = LeakyReLU(alpha=0.1,name='L14')(x)
    # drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(x)

    concat_p = concatenate([pool4,pool3_2],axis=-1)

    conv5_1 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(concat_p)
    conv5_1 = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv5_1_GN')(conv5_1)
    conv5_1 = LeakyReLU(alpha=0.1,name='L15')(conv5_1)
    conv5_1_se = GlobalAveragePooling2D()(conv5_1)
    conv5_1_se = Reshape([1,1,256])(conv5_1_se)
    fc5_se = Dense(128,activation='relu',kernel_initializer='he_normal')(conv5_1_se)
    fc5_se = Dense(256,activation='sigmoid',kernel_initializer='he_normal')(fc5_se)
    conv5_1_se_o = multiply([conv5_1,fc5_se])
    conv5_2 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv5_1_se_o)
    conv5_2 = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv5_2_GN')(conv5_2)
    conv5_2 = LeakyReLU(alpha=0.1,name='L16')(conv5_2)
    conv5_2_se = GlobalAveragePooling2D()(conv5_2)
    conv5_2_se = Reshape([1, 1, 256])(conv5_2_se)
    fc5_se = Dense(128,activation='relu',kernel_initializer='he_normal')(conv5_2_se)
    fc5_se = Dense(256,activation='sigmoid',kernel_initializer='he_normal')(fc5_se)
    conv5_2_se_o = multiply([conv5_2,fc5_se])




    merge6 = merge([conv4, (UpSampling2D(size=(2, 2))(conv5_2_se_o))], mode='concat', concat_axis=3)
    up6 = Conv2D(128, 2, padding='same', kernel_initializer='he_normal')(merge6)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv6_0_GN')(up6)
    x = LeakyReLU(alpha=0.1,name='uL17')(x)
    conv6 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv6_1_GN')(conv6)
    x = LeakyReLU(alpha=0.1,name='L17')(x)
    conv6 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv6_2_GN')(conv6)
    x = LeakyReLU(alpha=0.1,name='L18')(x)

    merge7 = merge([conv3, (UpSampling2D(size=(2, 2))(x))], mode='concat', concat_axis=3)
    up7 = Conv2D(64, 2, padding='same', kernel_initializer='he_normal')(merge7)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv7_0_GN')(up7)
    x = LeakyReLU(alpha=0.1,name='uL19')(x)
    conv7 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv7_1_GN')(conv7)
    x = LeakyReLU(alpha=0.1,name='L19')(x)
    conv7 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv7_2_GN')(conv7)
    x = LeakyReLU(alpha=0.1,name='L20')(x)

    merge8 = merge([conv2, (UpSampling2D(size=(2, 2))(x))], mode='concat', concat_axis=3)
    up8 = Conv2D(64, 2, padding='same', kernel_initializer='he_normal')(merge8)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv8_0_GN')(up8)
    x = LeakyReLU(alpha=0.1,name='uL21')(x)
    conv8 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv8_1_GN')(conv8)
    x = LeakyReLU(alpha=0.1,name='L21')(x)
    conv8 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv8_2_GN')(conv8)
    x = LeakyReLU(alpha=0.1,name='L22')(x)

    merge9 = merge([conv1, (UpSampling2D(size=(2, 2))(x))], mode='concat', concat_axis=3)
    up9 = Conv2D(32, 2, padding='same', kernel_initializer='he_normal')(merge9)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv9_0_GN')(up9)
    x = LeakyReLU(alpha=0.1,name='uL23')(x)
    conv9 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv9_1_GN')(conv9)
    x = LeakyReLU(alpha=0.1,name='L23')(x)
    conv9 = Conv2D(32, 3,  padding='same', kernel_initializer='he_normal')(x)
    # x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv9_2_GN')(conv9)
    # x = LeakyReLU(alpha=0.1,name='L24')(x)

    conv10 = Conv2D(1, 1,kernel_initializer=initializers.TruncatedNormal(mean=0.5,stddev=0.1),activation='sigmoid')(conv9)

    model = Model(input=[input1,input2], output=[conv5_1])

    # model.compile(optimizer=SGD(lr=1.0e-8), loss=EuclideanLoss, metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])

    return model

def threeDNet_multi_3D(input_shape=(None, None,None, 1)):

    input1 = Input(shape=(input_shape[0], input_shape[1],input_shape[2], 1), name='input1')
    input2 = Input(shape=(256, 256,input_shape[2], 1), name = 'input2')

    conv1 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(input1)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_1_GN')(conv1)
    x = LeakyReLU(alpha=0.1,name='L1')(x)
    conv1 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_2_GN')(conv1)
    x = LeakyReLU(alpha=0.1,name='L2')(x)
    pool1 = MaxPooling3D(pool_size=(2, 2,1))(x)
    print("pool1 shape:", pool1.shape)

    conv2 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(pool1)
    print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv2_1_GN')(conv2)
    x = LeakyReLU(alpha=0.1,name='L3')(x)
    conv2 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv2_2_GN')(conv2)
    x = LeakyReLU(alpha=0.1,name='L4')(x)
    pool2 = MaxPooling3D(pool_size=(2, 2,1))(x)
    print("pool2 shape:", pool2.shape)

    conv3 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(pool2)
    print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv3_1_GN')(conv3)
    x = LeakyReLU(alpha=0.1,name='L5')(x)
    conv3 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv3_2_GN')(conv3)
    x = LeakyReLU(alpha=0.1,name='L6')(x)
    pool3 = MaxPooling3D(pool_size=(2, 2,1))(x)
    print("pool3 shape:", pool3.shape)


    conv1_2 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(input2)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_1_GN2')(conv1_2)
    x = LeakyReLU(alpha=0.1,name='L7')(x)
    conv1_2 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_2_GN2')(conv1_2)
    x = LeakyReLU(alpha=0.1,name='L8')(x)
    pool1_2 = MaxPooling3D(pool_size=(2, 2,1))(x)
    print("pool1 shape:", pool1.shape)

    conv2_2 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(pool1_2)
    print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv2_1_GN2')(conv2_2)
    x = LeakyReLU(alpha=0.1,name='L9')(x)
    conv2_2 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv2_2_GN2')(conv2_2)
    x = LeakyReLU(alpha=0.1,name='L10')(x)
    pool2_2 = MaxPooling3D(pool_size=(2, 2,1))(x)
    print("pool2 shape:", pool2.shape)

    conv3_2 = Conv3D(128, 3, padding='same', kernel_initializer='he_normal')(pool2_2)
    print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv3_1_GN2')(conv3_2)
    x = LeakyReLU(alpha=0.1,name='L11')(x)
    conv3_2 = Conv3D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv3_2_GN2')(conv3_2)
    x = LeakyReLU(alpha=0.1,name='L12')(x)
    pool3_2 = MaxPooling3D(pool_size=(2, 2,1))(x)
    print("pool3 shape:", pool3.shape)


    conv4 = Conv3D(128, 3, padding='same', kernel_initializer='he_normal')(pool3)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv4_1_GN')(conv4)
    x = LeakyReLU(alpha=0.1,name='L13')(x)
    conv4 = Conv3D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv4_2_GN')(conv4)
    x = LeakyReLU(alpha=0.1,name='L14')(x)
    # drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2,1))(x)

    concat_p = concatenate([pool4,pool3_2],axis=-1)

    conv5 = Conv3D(256, 3, padding='same', kernel_initializer='he_normal')(concat_p)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv5_1_GN')(conv5)
    x = LeakyReLU(alpha=0.1,name='L15')(x)
    conv5 = Conv3D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv5_2_GN')(conv5)
    x = LeakyReLU(alpha=0.1,name='L16')(x)
    # drop5 = Dropout(0.5)(conv5)

    merge6 = merge([conv4, (UpSampling3D(size=(2, 2,1))(x))], mode='concat', concat_axis=-1)
    up6 = Conv3D(128, 2, padding='same', kernel_initializer='he_normal')(merge6)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv6_0_GN')(up6)
    x = LeakyReLU(alpha=0.1,name='uL17')(x)
    conv6 = Conv3D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv6_1_GN')(conv6)
    x = LeakyReLU(alpha=0.1,name='L17')(x)
    conv6 = Conv3D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv6_2_GN')(conv6)
    x = LeakyReLU(alpha=0.1,name='L18')(x)

    merge7 = merge([conv3, (UpSampling3D(size=(2, 2,1))(x))], mode='concat', concat_axis=-1)
    up7 = Conv3D(64, 2, padding='same', kernel_initializer='he_normal')(merge7)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv7_0_GN')(up7)
    x = LeakyReLU(alpha=0.1,name='uL19')(x)
    conv7 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv7_1_GN')(conv7)
    x = LeakyReLU(alpha=0.1,name='L19')(x)
    conv7 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv7_2_GN')(conv7)
    x = LeakyReLU(alpha=0.1,name='L20')(x)

    merge8 = merge([conv2, (UpSampling3D(size=(2, 2,1))(x))], mode='concat', concat_axis=-1)
    up8 = Conv3D(64, 2, padding='same', kernel_initializer='he_normal')(merge8)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv8_0_GN')(up8)
    x = LeakyReLU(alpha=0.1,name='uL21')(x)
    conv8 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv8_1_GN')(conv8)
    x = LeakyReLU(alpha=0.1,name='L21')(x)
    conv8 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv8_2_GN')(conv8)
    x = LeakyReLU(alpha=0.1,name='L22')(x)

    merge9 = merge([conv1, (UpSampling3D(size=(2, 2,1))(x))], mode='concat', concat_axis=-1)
    up9 = Conv3D(32, 2, padding='same', kernel_initializer='he_normal')(merge9)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv9_0_GN')(up9)
    x = LeakyReLU(alpha=0.1,name='uL23')(x)
    conv9 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv9_1_GN')(conv9)
    x = LeakyReLU(alpha=0.1,name='L23')(x)
    conv9 = Conv3D(32, 3,  padding='same', kernel_initializer='he_normal')(x)
    # x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv9_2_GN')(conv9)
    # x = LeakyReLU(alpha=0.1,name='L24')(x)

    conv10 = Conv3D(1, 1,kernel_initializer=initializers.TruncatedNormal(mean=0.5,stddev=0.1),activation='sigmoid')(conv9)

    model = Model(input=[input1,input2], output=[conv10])

    # model.compile(optimizer=SGD(lr=1.0e-8), loss=EuclideanLoss, metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])

    return model

def threeDNet_u2(input_shape=(None, None, 1)):

    input1 = Input(shape=(input_shape[0], input_shape[1], 1), name='input1')
    input2 = Input(shape=(input_shape[0], input_shape[1], 1),name = 'input2')

    input_concat = concatenate([input1,input2],axis=-1)

    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(input_concat)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_1_GN')(conv1)
    x = Activation('relu')(x)
    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_2_GN')(conv1)
    x = Activation('relu')(x)
    pool1 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool1 shape:", pool1.shape)

    conv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(pool1)
    print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv2_1_GN')(conv2)
    x = Activation('relu')(x)
    conv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv2_2_GN')(conv2)
    x = Activation('relu')(x)
    pool2 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool2 shape:", pool2.shape)

    conv3 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(pool2)
    print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv3_1_GN')(conv3)
    x = Activation('relu')(x)
    conv3 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv3_2_GN')(conv3)
    x = Activation('relu')(x)
    pool3 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool3 shape:", pool3.shape)

    conv4 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool3)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv4_1_GN')(conv4)
    x = Activation('relu')(x)
    conv4 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv4_2_GN')(conv4)
    x = Activation('relu')(x)
    # drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(x)

    conv5 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool4)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv5_1_GN')(conv5)
    x = Activation('relu')(x)
    conv5 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv5_2_GN')(conv5)
    x = Activation('relu')(x)
    # drop5 = Dropout(0.5)(conv5)

    merge6 = merge([conv4, (UpSampling2D(size=(2, 2))(x))], mode='concat', concat_axis=3)
    up6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(up6)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv6_1_GN')(conv6)
    x = Activation('relu')(x)
    conv6 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv6_2_GN')(conv6)
    x = Activation('relu')(x)

    merge7 = merge([conv3, (UpSampling2D(size=(2, 2))(x))], mode='concat', concat_axis=3)
    up7 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(up7)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv7_1_GN')(conv7)
    x = Activation('relu')(x)
    conv7 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv7_2_GN')(conv7)
    x = Activation('relu')(x)

    merge8 = merge([conv2, (UpSampling2D(size=(2, 2))(x))], mode='concat', concat_axis=3)
    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(up8)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv8_1_GN')(conv8)
    x = Activation('relu')(x)
    conv8 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv8_2_GN')(conv8)
    x = Activation('relu')(x)

    merge9 = merge([conv1, (UpSampling2D(size=(2, 2))(x))], mode='concat', concat_axis=3)
    up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(up9)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv9_1_GN')(conv9)
    x = Activation('relu')(x)
    conv9 = Conv2D(32, 3,  padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv9_2_GN')(conv9)
    x = Activation('relu')(x)

    # output_concat = concatenate([x,input2],axis=-1)
    # conv10 = Conv2D(2, 3,kernel_initializer='he_normal',padding ='same',name = 'conv10')(output_concat)
    # x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv10_1_GN')(conv10)
    # x = Activation('relu')(x)

    out = Conv2D(1, 1,kernel_initializer=initializers.TruncatedNormal(mean=0.5,stddev=0.5),activation='sigmoid')(x)

    model = Model(input=[input1,input2], output=[out])

    # model.compile(optimizer = Adam(lr=1e-6), loss=['binary_crossentropy'],metrics=['acc',EuclideanLoss,precision,recall,fmeasure,yt_sum])

    return model

def threeDNet_multi_feature_extrator(input_shape=(None, None, 1)):
    input1 = Input(shape=(input_shape[0], input_shape[1], 1), name='input1')
    input2 = Input(shape=(256, 256, 1), name='input2')

    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(input1)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_1_GN')(conv1)
    x = LeakyReLU(alpha=0.1, name='L1')(x)
    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_2_GN')(conv1)
    x = LeakyReLU(alpha=0.1, name='L2')(x)
    pool1 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool1 shape:", pool1.shape)

    conv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(pool1)
    print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv2_1_GN')(conv2)
    x = LeakyReLU(alpha=0.1, name='L3')(x)
    conv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv2_2_GN')(conv2)
    x = LeakyReLU(alpha=0.1, name='L4')(x)
    pool2 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool2 shape:", pool2.shape)

    conv3 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(pool2)
    print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv3_1_GN')(conv3)
    x = LeakyReLU(alpha=0.1, name='L5')(x)
    conv3 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv3_2_GN')(conv3)
    x = LeakyReLU(alpha=0.1, name='L6')(x)
    pool3 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool3 shape:", pool3.shape)

    conv1_2 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(input2)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_1_GN2')(conv1_2)
    x = LeakyReLU(alpha=0.1, name='L7')(x)
    conv1_2 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv1_2_GN2')(conv1_2)
    x = LeakyReLU(alpha=0.1, name='L8')(x)
    pool1_2 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool1 shape:", pool1.shape)

    conv2_2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(pool1_2)
    print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv2_1_GN2')(conv2_2)
    x = LeakyReLU(alpha=0.1, name='L9')(x)
    conv2_2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv2_2_GN2')(conv2_2)
    x = LeakyReLU(alpha=0.1, name='L10')(x)
    pool2_2 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool2 shape:", pool2.shape)

    conv3_2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool2_2)
    print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv3_1_GN2')(conv3_2)
    x = LeakyReLU(alpha=0.1, name='L11')(x)
    conv3_2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv3_2_GN2')(conv3_2)
    x = LeakyReLU(alpha=0.1, name='L12')(x)
    pool3_2 = MaxPooling2D(pool_size=(2, 2))(x)
    print("pool3 shape:", pool3.shape)

    conv4 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool3)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv4_1_GN')(conv4)
    x = LeakyReLU(alpha=0.1, name='L13')(x)
    conv4 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv4_2_GN')(conv4)
    x = LeakyReLU(alpha=0.1, name='L14')(x)
    # drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(x)

    concat_p = concatenate([pool4, pool3_2], axis=-1)

    conv5 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(concat_p)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv5_1_GN')(conv5)
    x = LeakyReLU(alpha=0.1, name='L15')(x)
    conv5 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='entry_flow_conv5_2_GN')(conv5)
    x = LeakyReLU(alpha=0.1, name='L16')(x)
    # drop5 = Dropout(0.5)(conv5)

    model = Model(input=[input1, input2], output=[x])

    # model.compile(optimizer=SGD(lr=1.0e-8), loss=EuclideanLoss, metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])

    return model

def lstm_3D_net_seg(input_shape=(None,None,None,1),lstm_block = 2,filters_num=45):

    lstm_block_num = lstm_block
    # filters_num = 45
    input = Input(shape=(input_shape[0], input_shape[1], input_shape[2],input_shape[3]), name='input1')

    for n in range(0,lstm_block_num):
        x = ConvLSTM2D(filters=filters_num,kernel_size=(3,3),padding='same',name='convlstm2D_'+str(n),return_sequences=True)(input)
        # print('convlstm2D_',str(n),"shape:", x.shape)
        x = GroupNormalization(groups=2, axis=2, epsilon=0.1, name='GN_'+str(n))(x)
        x = LeakyReLU(alpha=0.1, name='Leakyrelu' + str(n))(x)

    # x = MaxPooling3D(pool_size=(1,2,2),name='maxpooling_lstm')(x)

    # x = ConvLSTM2D(filters=32, kernel_size=(1, 1), padding='same', name='convlstm2D_' + str(lstm_block_num),
    #                return_sequences=True)(x)
    # # print('convlstm2D_',str(n),"shape:", x.shape)
    # x = GroupNormalization(groups=2, axis=2, epsilon=0.1, name='GN_' + str(lstm_block_num))(x)
    # x = LeakyReLU(alpha=0.1, name='Leakyrelu' + str(lstm_block_num))(x)

    out = ConvLSTM2D(filters=1, kernel_size=(1, 1), activation='sigmoid',padding='same', name='convlstm2D_out',return_sequences=True)(x)

    # x = Conv2DTranspose(filters=filters_num,kernel_size=4,strides=2,padding='same',name='deconv1')(x)

    # out = Conv2D(filters=1,kernel_size=1,kernel_initializer=initializers.TruncatedNormal(mean=0.5,stddev=0.1),activation='sigmoid',name='conv_out')(x)

    # x = MaxPooling2D(pool_size=(2,2))(x)
    #
    # for n in range(1,lstm_block_num+1):
    #     x = ConvLSTM2D(filters=filters_num*2,kernel_size=(3,3),padding='same',name='convlstm2D_'+str(n),return_sequences=True)(x)
    #     x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='GN_'+str(n))(x)
    #     x = LeakyReLU(alpha=0.1, name='Leakyrelu' + str(n))(x)
    #
    # x = Conv2DTranspose(kernel_size=(4,4),strides=(2,2),padding='valid',name = )


    model = Model(input=[input], output=[out])
    return model

def lstm_3D_net_seg_no_return_seq(input_shape=(None,None,None,1),lstm_block = 2,filters_num=45):

    lstm_block_num = lstm_block
    # filters_num = 45
    input = Input(shape=(input_shape[0], input_shape[1], input_shape[2],input_shape[3]), name='input1')

    for n in range(0,lstm_block_num):
        x = ConvLSTM2D(filters=filters_num,kernel_size=(3,3),padding='same',name='convlstm2D_'+str(n),return_sequences=True)(input)
        # print('convlstm2D_',str(n),"shape:", x.shape)
        x = GroupNormalization(groups=2, axis=2, epsilon=0.1, name='GN_'+str(n))(x)
        x = LeakyReLU(alpha=0.1, name='Leakyrelu' + str(n))(x)


    x = MaxPooling3D(pool_size=(1,2,2),name='maxpooling_lstm')(x)

    x = ConvLSTM2D(filters=filters_num, kernel_size=(3, 3), padding='same', name='convlstm2D_' + str(lstm_block_num),
                   return_sequences=False)(x)
    # print('convlstm2D_',str(n),"shape:", x.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='GN_' + str(lstm_block_num))(x)
    x = LeakyReLU(alpha=0.1, name='Leakyrelu' + str(lstm_block_num))(x)

    x = Conv2DTranspose(filters=filters_num,kernel_size=4,strides=2,padding='same',name='deconv1')(x)
    x = Conv2D(filters=filters_num,kernel_size=3,kernel_initializer='he_normal',padding='same',name='conv_1')(x)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='GN_c1')(x)
    x = LeakyReLU(alpha=0.1, name='Leakyrelu_c1')(x)
    out = Conv2D(filters=filters_num,kernel_size=1,kernel_initializer=initializers.TruncatedNormal(mean=0.5,stddev=0.1),activation='sigmoid',name='conv_out')(x)


    model = Model(input=[input], output=[out])
    return model

def lstm_3D_net_seg_feature_no_return_seq(input_shape=(None,None,None,256),lstm_block = 2,filters_num=45):

    lstm_block_num = lstm_block
    # filters_num = 45
    input = Input(shape=(input_shape[0], input_shape[1], input_shape[2],input_shape[3]), name='input1')

    for n in range(0,lstm_block_num):
        x = ConvLSTM2D(filters=filters_num,kernel_size=(3,3),padding='same',name='convlstm2D_'+str(n),return_sequences=True)(input)
        # print('convlstm2D_',str(n),"shape:", x.shape)
        x = GroupNormalization(groups=2, axis=2, epsilon=0.1, name='GN_'+str(n))(x)
        x = LeakyReLU(alpha=0.1, name='Leakyrelu' + str(n))(x)


    x = ConvLSTM2D(filters=filters_num, kernel_size=(3, 3), padding='same', name='convlstm2D_' + str(lstm_block_num),
                   return_sequences=False)(x)
    # print('convlstm2D_',str(n),"shape:", x.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='GN_' + str(lstm_block_num))(x)
    x = LeakyReLU(alpha=0.1, name='Leakyrelu' + str(lstm_block_num))(x)

    for n in range(0,4):
        x = Conv2DTranspose(filters=filters_num,kernel_size=4,strides=2,padding='same',name='deconv_'+str(n))(x)
        x = Conv2D(filters=filters_num,kernel_size=3,kernel_initializer='he_normal',padding='same',name='conv_'+str(n))(x)
        x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='GN_c_'+str(n))(x)
        x = LeakyReLU(alpha=0.1, name='Leakyrelu_c_'+str(n))(x)

    out = Conv2D(filters=filters_num,kernel_size=1,kernel_initializer=initializers.TruncatedNormal(mean=0.5,stddev=0.1),activation='sigmoid',name='conv_out')(x)


    model = Model(input=[input], output=[out])
    return model

def lstm_3D_net_predict(input_shape=(None,None,None,1),lstm_block = 2,filters_num=31):

    lstm_block_num = lstm_block
    # filters_num = 45
    input = Input(shape=(input_shape[0], input_shape[1], input_shape[2],input_shape[3]), name='input1')

    for n in range(0,lstm_block_num):
        x = ConvLSTM2D(filters=filters_num,kernel_size=(3,3),padding='same',name='convlstm2D_'+str(n),return_sequences=True)(input)
        # print('convlstm2D_',str(n),"shape:", x.shape)
        x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='GN_'+str(n))(x)
        x = LeakyReLU(alpha=0.1, name='Leakyrelu' + str(n))(x)

    x = MaxPooling3D(pool_size=(1,4,4),name='maxpooling_lstm')(x)

    x = ConvLSTM2D(filters=12, kernel_size=(3, 3), padding='same', name='convlstm2D_' + str(lstm_block_num),
                   return_sequences=True)(x)
    # print('convlstm2D_',str(n),"shape:", x.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='GN_' + str(lstm_block_num))(x)
    x = LeakyReLU(alpha=0.1, name='Leakyrelu' + str(lstm_block_num))(x)

    x = ConvLSTM2D(filters=12, kernel_size=(3, 3), padding='same', name='convlstm2D_' + str(lstm_block_num+1),
                   return_sequences=True)(x)
    # print('convlstm2D_',str(n),"shape:", x.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='GN_' + str(lstm_block_num+1))(x)
    x = LeakyReLU(alpha=0.1, name='Leakyrelu' + str(lstm_block_num+1))(x)

    x = UpSampling3D(size=(1,4,4),name = 'UpSampling_lstm')(x)

    x = ZeroPadding3D(padding=(0,1,1))(x)
    x = ConvLSTM2D(filters=6, kernel_size=(2, 2), padding='valid', name='convlstm2D_' + str(lstm_block_num+2),
                   return_sequences=True)(x)
    # print('convlstm2D_',str(n),"shape:", x.shape)
    x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='GN_' + str(lstm_block_num+2))(x)
    x = LeakyReLU(alpha=0.1, name='Leakyrelu' + str(lstm_block_num+2))(x)

    out = ConvLSTM2D(filters=1, kernel_size=(1, 1), activation='sigmoid',kernel_initializer='he_normal'
                     ,padding='same', name='convlstm2D_out',return_sequences=True)(x)

    # x = Conv2DTranspose(filters=filters_num,kernel_size=4,strides=2,padding='same',name='deconv1')(x)

    # out = Conv2D(filters=30,kernel_size=1,kernel_initializer=initializers.TruncatedNormal(mean=0.8,stddev=0.1),activation='sigmoid',name='conv_out')(x)

    # x = MaxPooling2D(pool_size=(2,2))(x)
    #
    # for n in range(1,lstm_block_num+1):
    #     x = ConvLSTM2D(filters=filters_num*2,kernel_size=(3,3),padding='same',name='convlstm2D_'+str(n),return_sequences=True)(x)
    #     x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='GN_'+str(n))(x)
    #     x = LeakyReLU(alpha=0.1, name='Leakyrelu' + str(n))(x)
    #
    # x = Conv2DTranspose(kernel_size=(4,4),strides=(2,2),padding='valid',name = )


    model = Model(input=[input], output=[out])
    return model

def lstm_3D_net_predict_2(input_shape=(None,None,None,1),lstm_block = 2,filters_num=31):

    lstm_block_num = lstm_block
    # filters_num = 45
    input = Input(shape=(input_shape[0], input_shape[1], input_shape[2],input_shape[3]), name='input1')

    for n in range(0,lstm_block_num):
        x = ConvLSTM2D(filters=filters_num,kernel_size=(3,3),activation='relu',padding='same',name='convlstm2D_'+str(n),return_sequences=True)(input)
        # print('convlstm2D_',str(n),"shape:", x.shape)
        # x = GroupNormalization(groups=2, axis=2, epsilon=0.1, name='GN_'+str(n))(x)
        # x = LeakyReLU(alpha=0.1, name='Leakyrelu' + str(n))(x)

    # x = MaxPooling3D(pool_size=(1, 4, 4), name='maxpooling_lstm')(x)

    x = ConvLSTM2D(filters=12, kernel_size=(3, 3), padding='same',activation='relu', name='convlstm2D_' + str(lstm_block_num),
                   return_sequences=True)(x)
    # print('convlstm2D_',str(n),"shape:", x.shape)
    # x = GroupNormalization(groups=2, axis=2, epsilon=0.1, name='GN_' + str(lstm_block_num))(x)
    # x = LeakyReLU(alpha=0.1, name='Leakyrelu' + str(lstm_block_num))(x)

    # x = ConvLSTM2D(filters=12, kernel_size=(3, 3), padding='same',activation='relu', name='convlstm2D_' + str(lstm_block_num+1),
    #                return_sequences=False)(x)
    # print('convlstm2D_',str(n),"shape:", x.shape)
    # x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='GN_' + str(lstm_block_num+1))(x)
    # x = LeakyReLU(alpha=0.1, name='Leakyrelu' + str(lstm_block_num+1))(x)

    x = ConvLSTM2D(filters=6, kernel_size=(3, 3), padding='same',activation='relu', name='convlstm2D_' + str(lstm_block_num+2),
                   return_sequences=False)(x)
    # print('convlstm2D_',str(n),"shape:", x.shape)
    # x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='GN_' + str(lstm_block_num+2))(x)
    # x = LeakyReLU(alpha=0.1, name='Leakyrelu' + str(lstm_block_num+2))(x)

    out = Conv2D(filters=6, kernel_size=(1, 1), activation='sigmoid',kernel_initializer='he_normal'
                     ,padding='same', name='conv2D_out')(x)

    # x = Conv2DTranspose(filters=filters_num,kernel_size=4,strides=2,padding='same',name='deconv1')(x)

    # out = Conv2D(filters=30,kernel_size=1,kernel_initializer=initializers.TruncatedNormal(mean=0.8,stddev=0.1),activation='sigmoid',name='conv_out')(x)

    # x = MaxPooling2D(pool_size=(2,2))(x)
    #
    # for n in range(1,lstm_block_num+1):
    #     x = ConvLSTM2D(filters=filters_num*2,kernel_size=(3,3),padding='same',name='convlstm2D_'+str(n),return_sequences=True)(x)
    #     x = GroupNormalization(groups=2, axis=1, epsilon=0.1, name='GN_'+str(n))(x)
    #     x = LeakyReLU(alpha=0.1, name='Leakyrelu' + str(n))(x)
    #
    # x = Conv2DTranspose(kernel_size=(4,4),strides=(2,2),padding='valid',name = )


    model = Model(input=[input], output=[out])
    return model


def Vgg_base_layers():
    base_model = ResNet50(weights='imagenet', include_top=False)
    # model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_conv3').output)

    return base_model

def lstm_input(lstm_layer=5):

    input1 = Input(shape=(lstm_layer,None, None, 1), name='input1')
    input2 = Input(shape=(None, None, 1), name='input2')
    x_l = ConvLSTM2D(filters=lstm_layer, kernel_size=(3, 3), activation='relu', padding='same',name='convlstm2D_1', return_sequences=False)(input1)
    x_c = Conv2D(3,3,activation='relu',padding='same',name='conv1')(input2)
    con_lc = concatenate([x_c,x_l],axis=-1)
    x = Conv2D(3,3,activation='relu',padding='same',name='conv2')(con_lc)
    model = Model(input=[input1,input2], output=[x])
    return model

def cls():
    inputs = Input(shape=(None, None, 2048), name='inputs')
    x = GlobalMaxPooling2D()(inputs)
    # fla = Flatten()(inputs)
    # fc1 = Dense(256,activation='relu')(inputs)
    # fc2 = Dense(128, activation='relu')(fc1)
    # fc3 = Dense(64, activation='relu')(fc2)
    fc4 = Dense(2,activation='softmax')(x)
    model = Model(input=[inputs], output=[fc4])
    return model

def vgg_cls(vgg, cls):
    input1 = Input(shape=(None, None, 3))
    result_vgg = vgg(input1)
    # d_m.trainable = False
    result_cls = cls(result_vgg)
    # result_rpn_regr = rpn_m(result_vgg)

    model = Model(input=[input1], output=[result_cls])
    return model

def lstm_vgg_cls(lstm,vgg, cls):
    input1 = Input(shape=(5,None, None, 1))
    input2 = Input(shape=(None, None, 1))
    result_lstm = lstm([input1,input2])
    result_vgg = vgg(result_lstm)
    # d_m.trainable = False
    result_cls = cls(result_vgg)
    # result_rpn_regr = rpn_m(result_vgg)

    model = Model(input=[input1,input2], output=[result_cls])
    return model

def threeDNet_single_SE(input_shape=(None, None, 1),gn_axis = 3,GN_able = 1,GN_grou = 2,lk_alph = 0.1):

    input1 = Input(shape=(input_shape[0], input_shape[1], 1), name='input1')
    # input2 = Input(shape=(256, 256, 1), name = 'input2')

    c1 = conv_gn_leaky(input1, 3, 64, stage=1, block='a',gn_able = GN_able,gn_group=GN_grou,leaky_alpha = lk_alph, strides=(1, 1))
    c1 = conv_gn_leaky(c1, 3, 64, stage=1, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph, strides=(1, 1))
    pool1 = MaxPooling2D(pool_size=(2, 2))(c1)

    c2 = conv_gn_leaky(pool1, 3, 128, stage=2, block='a',gn_able = GN_able,gn_group=GN_grou,leaky_alpha = lk_alph, strides=(1, 1))
    c2 = conv_gn_leaky(c2, 3, 128, stage=2, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph, strides=(1, 1))
    pool2 = MaxPooling2D(pool_size=(2, 2))(c2)
    #240*80
    c3 = conv_gn_leaky(pool2, 3, 256, stage=3, block='a',gn_able = GN_able,gn_group=GN_grou,leaky_alpha = lk_alph, strides=(1, 1))
    c3 = conv_gn_leaky(c3, 3, 256, stage=3, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph, strides=(1, 1))
    pool3 = MaxPooling2D(pool_size=(2, 2))(c3)


    c4 = conv_gn_leaky(pool3, 3, 512, stage=4, block='a',gn_able = GN_able,gn_group=GN_grou,leaky_alpha = lk_alph, strides=(1, 1))
    c4 = conv_gn_leaky(c4, 3, 512, stage=4, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph, strides=(1, 1))
    pool4 = MaxPooling2D(pool_size=(2, 2))(c4)

    # c4_2 = conv_gn_leaky(pool4, 3, 512, stage=4, block='a', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
    #                    strides=(1, 1))
    # c4_2 = conv_gn_leaky(c4_2, 3, 512, stage=4, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
    #                    strides=(1, 1))
    # pool4_2 = MaxPooling2D(pool_size=(2, 2))(c4_2)

    # concat_p = concatenate([pool4,pool3_2],axis=-1)

    c5_1 = conv_gn_leaky(pool4, 3, 1024, stage=5, block='a', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c5 = squeeze_excite_block(c5_1,ratio=8)
    c5 = conv_gn_leaky(c5, 3, 1024, stage=5, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c5 = squeeze_excite_block(c5, ratio=8)

    # print('convc4:',c4.shape)
    # print('conv5:', c5.shape)

    merge6 = concatenate([c4, (UpSampling2D(size=(2, 2))(c5))], axis=-1)
    up6 = conv_gn_leaky(merge6, 3, 512, stage=6, block='a', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c6 = conv_gn_leaky(up6, 3, 512, stage=6, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c6 = conv_gn_leaky(c6, 3, 256, stage=6, block='c', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))

    merge7 = concatenate([c3, (UpSampling2D(size=(2, 2))(c6))], axis=-1)
    up7 = conv_gn_leaky(merge7, 3, 256, stage=7, block='a', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c7 = conv_gn_leaky(up7, 3, 256, stage=7, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c7 = conv_gn_leaky(c7, 3,128, stage=7, block='c', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))

    merge8 = concatenate([c2, (UpSampling2D(size=(2, 2))(c7))], axis=-1)
    up8 = conv_gn_leaky(merge8, 3, 128, stage=8, block='a', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c8 = conv_gn_leaky(up8, 3, 128, stage=8, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c8 = conv_gn_leaky(c8, 3, 64, stage=8, block='c', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))

    merge9 = concatenate([c1, (UpSampling2D(size=(2, 2))(c8))], axis=-1)
    up9 = conv_gn_leaky(merge9, 3, 64, stage=9, block='a', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c9 = conv_gn_leaky(up9, 3, 64, stage=9, block='b', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))
    c9 = conv_gn_leaky(c9, 3, 64, stage=9, block='c', gn_able=GN_able, gn_group=GN_grou, leaky_alpha=lk_alph,
                      strides=(1, 1))

    conv10 = Conv2D(1, 1,kernel_initializer=initializers.TruncatedNormal(mean=0.5,stddev=0.1),activation='sigmoid')(c9)

    model = Model(input=[input1], output=[conv10])

    # model.compile(optimizer=SGD(lr=1.0e-8), loss=EuclideanLoss, metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])

    return model

def threeDNet_ycl(input_shape=(None, None, 1)):

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

def threeDNet_ycl_SE(input_shape=(None, None, 1)):

    input1 = Input(shape=(input_shape[0], input_shape[1], 1), name='input1')

    conv1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(input1)
    print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv1_1_GN')(conv1)
    x = Activation('relu')(x)
    x = squeeze_excite_block(x, ratio=8)
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
    x = squeeze_excite_block(x, ratio=8)
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
    x = squeeze_excite_block(x, ratio=8)
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

    model = Model(input=[input1], output=[conv10])

    # model.compile(optimizer=SGD(lr=1.0e-8), loss=EuclideanLoss, metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])

    return model

def threeDNet_ycl_small(input_shape=(None, None, 1)):

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
    # pool4 = MaxPooling2D(pool_size=(2, 2))(x)
    #
    # conv5 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(pool4)
    # x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv5_1_GN')(conv5)
    # x = Activation('relu')(x)
    # conv5 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(x)
    # x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv5_2_GN')(conv5)
    # x = Activation('relu')(x)
    # drop5 = Dropout(0.5)(conv5)

    merge6 = concatenate([conv3, (UpSampling2D(size=(2, 2))(x))], axis=3)
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(up6)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv6_1_GN')(conv6)
    x = Activation('relu')(x)
    conv6 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv6_2_GN')(conv6)
    x = Activation('relu')(x)

    merge7 = concatenate([conv2, (UpSampling2D(size=(2, 2))(x))], axis=3)
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(up7)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv7_1_GN')(conv7)
    x = Activation('relu')(x)
    conv7 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv7_2_GN')(conv7)
    x = Activation('relu')(x)

    merge8 = concatenate([conv1, (UpSampling2D(size=(2, 2))(x))],axis=3)
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(up8)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv8_1_GN')(conv8)
    x = Activation('relu')(x)
    conv8 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv8_2_GN')(conv8)
    x = Activation('relu')(x)
    #
    # merge9 = concatenate([conv1, (UpSampling2D(size=(2, 2))(x))], axis=3)
    # up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    # conv9 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(up9)
    # x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv9_1_GN')(conv9)
    # x = Activation('relu')(x)
    # conv9 = Conv2D(64, 3,  padding='same', kernel_initializer='he_normal')(x)
    # x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv9_2_GN')(conv9)
    # x = Activation('relu')(x)

    conv10 = Conv2D(1, 1,kernel_initializer=initializers.TruncatedNormal(mean=0.2,stddev=0.1),activation='sigmoid')(x)

    model = Model(input=[input1], output=[conv10])

    # model.compile(optimizer=SGD(lr=1.0e-8), loss=EuclideanLoss, metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])

    return model

def twoDNet_xxt(input_shape=(None, None, 1)):

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
def twoD_xxt_small(input_shape=(None, None, 1)):
    input1 = Input(shape=(input_shape[0], input_shape[1], 1), name='input1')

    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(input1)
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

    merge8 = concatenate([conv2, (UpSampling2D(size=(2, 2))(x))], axis=3)
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
    conv9 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv9_2_GN')(conv9)
    x = Activation('relu')(x)

    conv10 = Conv2D(1, 1, kernel_initializer=initializers.TruncatedNormal(mean=0.2, stddev=0.1), activation='sigmoid')(
        x)

    print('conv10(outputs): ', conv10.shape)
    model = Model(input=[input1], output=[conv10])

    # model.compile(optimizer=SGD(lr=1.0e-8), loss=EuclideanLoss, metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])

    return model

def threeDNet_ycl_small_2(input_shape=(None, None, 1)):

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
    # pool4 = MaxPooling2D(pool_size=(2, 2))(x)
    #
    # conv5 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(pool4)
    # x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv5_1_GN')(conv5)
    # x = Activation('relu')(x)
    # conv5 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(x)
    # x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv5_2_GN')(conv5)
    # x = Activation('relu')(x)
    # drop5 = Dropout(0.5)(conv5)

    merge6 = concatenate([conv3, (UpSampling2D(size=(2, 2))(x))], axis=3)
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(up6)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv6_1_GN')(conv6)
    x = Activation('relu')(x)
    conv6 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv6_2_GN')(conv6)
    x = Activation('relu')(x)

    merge7 = concatenate([conv2, (UpSampling2D(size=(2, 2))(x))], axis=3)
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(up7)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv7_1_GN')(conv7)
    x = Activation('relu')(x)
    conv7 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv7_2_GN')(conv7)
    x = Activation('relu')(x)

    merge8 = concatenate([conv1, (UpSampling2D(size=(2, 2))(x))],axis=3)
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(up8)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv8_1_GN')(conv8)
    x = Activation('relu')(x)
    conv8 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv8_2_GN')(conv8)
    x = Activation('relu')(x)
    #
    # merge9 = concatenate([conv1, (UpSampling2D(size=(2, 2))(x))], axis=3)
    # up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    # conv9 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(up9)
    # x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv9_1_GN')(conv9)
    # x = Activation('relu')(x)
    # conv9 = Conv2D(64, 3,  padding='same', kernel_initializer='he_normal')(x)
    # x = GroupNormalization(groups=2, axis=-1, epsilon=0.1, name='entry_flow_conv9_2_GN')(conv9)
    # x = Activation('relu')(x)

    conv10 = Conv2D(1, 1,kernel_initializer=initializers.TruncatedNormal(mean=0.2,stddev=0.1),activation='sigmoid')(x)

    model = Model(input=[input1], output=[conv10])

    # model.compile(optimizer=SGD(lr=1.0e-8), loss=EuclideanLoss, metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])

    return model

def DeeplabV2(input_shape=(None, None, 1)):
    # 输入命名为 input1，大小为512*512 的输入层 input_1
    # shape设置为3阶(维)
    img_input =Input(shape=(input_shape[0], input_shape[1], 1), name='input1')
    # Block 1
    h = ZeroPadding2D(padding=(1, 1))(img_input)
    h = Convolution2D(64, 3, 3, activation='relu', name='conv1_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(64, 3, 3, activation='relu', name='conv1_2')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

    # Block 2
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_2')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

    # Block 3
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_2')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_3')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

    # Block 4
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_2')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_3')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(h)

    # Block 5
    h = ZeroPadding2D(padding=(2, 2))(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_1')(h)
    h = ZeroPadding2D(padding=(2, 2))(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_2')(h)
    h = ZeroPadding2D(padding=(2, 2))(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_3')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    p5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(h)

    # branching for Atrous Spatial Pyramid Pooling
    # hole = 6
    b1 = ZeroPadding2D(padding=(6, 6))(p5)
    b1 = AtrousConvolution2D(1024, 3, 3, atrous_rate=(6, 6), activation='relu', name='fc6_1')(b1)
    b1 = Dropout(0.5)(b1)
    b1 = Convolution2D(1024, 1, 1, activation='relu', name='fc7_1')(b1)
    b1 = Dropout(0.5)(b1)
    b1 = Convolution2D(1, 1, 1, activation='relu', name='fc8_voc12_1')(b1)

    # hole = 12
    b2 = ZeroPadding2D(padding=(12, 12))(p5)
    b2 = AtrousConvolution2D(1024, 3, 3, atrous_rate=(12, 12), activation='relu', name='fc6_2')(b2)
    b2 = Dropout(0.5)(b2)
    b2 = Convolution2D(1024, 1, 1, activation='relu', name='fc7_2')(b2)
    b2 = Dropout(0.5)(b2)
    b2 = Convolution2D(1, 1, 1, activation='relu', name='fc8_voc12_2')(b2)

    # hole = 18
    b3 = ZeroPadding2D(padding=(18, 18))(p5)
    b3 = AtrousConvolution2D(1024, 3, 3, atrous_rate=(18, 18), activation='relu', name='fc6_3')(b3)
    b3 = Dropout(0.5)(b3)
    b3 = Convolution2D(1024, 1, 1, activation='relu', name='fc7_3')(b3)
    b3 = Dropout(0.5)(b3)
    b3 = Convolution2D(1, 1, 1, activation='relu', name='fc8_voc12_3')(b3)

    # hole = 24
    b4 = ZeroPadding2D(padding=(24, 24))(p5)
    b4 = AtrousConvolution2D(1024, 3, 3, atrous_rate=(24, 24), activation='relu', name='fc6_4')(b4)
    b4 = Dropout(0.5)(b4)
    b4 = Convolution2D(1024, 1, 1, activation='relu', name='fc7_4')(b4)
    b4 = Dropout(0.5)(b4)
    b4 = Convolution2D(1, 1, 1, activation='relu', name='fc8_voc12_4')(b4)

    # s = merge([b1, b2, b3, b4], mode='sum')
    s = add([b1, b2, b3, b4])
    logits = UpSampling2D(size=(8,8))(s)
    bilinear_layer = Conv2D(filters=1, kernel_size=(8, 8), strides=(1, 1),
                            activation='linear',
                            padding='same', name='conv', use_bias=False,
                            weights=bilinear_kernel(8, 8, 1, False))(logits)

    # out = bilinear_layer
    out = Convolution2D(1, 1, 1, activation='sigmoid',name='conv_out2')(bilinear_layer)

    # compile(self, optimizer, loss, metrics=None, sample_weight_mode=None)编译用来配置模型的学习过程
    # optimizer：字符串（预定义优化器名）或优化器对象,优化器是编译Keras模型必要的两个参数之一
                # 可以在调用model.compile()之前初始化一个优化器对象，然后传入该函数，如 sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
                                                                             # model.compile(loss='mean_squared_error', optimizer=sgd)
                # 也可以在调用model.compile()时传递一个预定义优化器名。在后者情形下，优化器的参数将使用默认值。model.compile(loss='mean_squared_error', optimizer='sgd')
    # loss：字符串（预定义损失函数名）或目标函数
    # metrics：列表，包含评估模型在训练和测试时的网络性能的指标，典型用法 metrics=['accuracy']
    model = Model(input=[img_input], output=out, name='deeplabV2')

    # model.compile(optimizer = Adam(lr = 1e-5), loss = ['binary_crossentropy'],metrics=['acc',EuclideanLoss,precision,recall,fmeasure,yt_sum])

    return model
