import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import SimpleITK as sitk
import os
from time import sleep

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import models

#os, time 내장
#tf도 내장인대 1.버전

#ID = os.environ['ID']

TRAIN_DIR_IMG = '/data/train/img/'
TRAIN_DIR_MASK = '/data/train/groundtruth/'
LOG_DIR = '/data/volume/logs'

name_list = ['WBCT_0001.img', 'WBCT_0002.img', 'WBCT_0003.img', 'WBCT_0004.img', 'WBCT_0005.img', 'WBCT_0006.img', 'WBCT_0007.img', 'WBCT_0008.img', 'WBCT_0009.img', 'WBCT_0010.img', 'WBCT_0011.img', 'WBCT_0012.img', 'WBCT_0013.img', 'WBCT_0014.img', 'WBCT_0015.img', 'WBCT_0016.img', 'WBCT_0017.img', 'WBCT_0018.img', 'WBCT_0019.img', 'WBCT_0020.img', 'WBCT_0021.img', 'WBCT_0022.img', 'WBCT_0023.img', 'WBCT_0024.img', 'WBCT_0025.img', 'WBCT_0026.img', 'WBCT_0027.img', 'WBCT_0028.img', 'WBCT_0029.img', 'WBCT_0030.img', 'WBCT_0031.img', 'WBCT_0032.img', 'WBCT_0033.img', 'WBCT_0034.img', 'WBCT_0035.img', 'WBCT_0036.img', 'WBCT_0037.img', 'WBCT_0038.img', 'WBCT_0039.img', 'WBCT_0040.img', 'WBCT_0041.img', 'WBCT_0042.img', 'WBCT_0043.img', 'WBCT_0044.img', 'WBCT_0045.img', 'WBCT_0046.img', 'WBCT_0047.img', 'WBCT_0048.img', 'WBCT_0049.img', 'WBCT_0050.img', 'WBCT_0051.img', 'WBCT_0052.img', 'WBCT_0053.img', 'WBCT_0054.img', 'WBCT_0055.img', 'WBCT_0056.img', 'WBCT_0057.img', 'WBCT_0058.img', 'WBCT_0059.img', 'WBCT_0060.img', 'WBCT_0061.img', 'WBCT_0062.img', 'WBCT_0063.img', 'WBCT_0064.img', 'WBCT_0065.img', 'WBCT_0066.img', 'WBCT_0067.img', 'WBCT_0068.img', 'WBCT_0069.img', 'WBCT_0070.img', 'WBCT_0071.img', 'WBCT_0072.img', 'WBCT_0073.img', 'WBCT_0074.img', 'WBCT_0075.img', 'WBCT_0076.img', 'WBCT_0077.img', 'WBCT_0078.img', 'WBCT_0079.img', 'WBCT_0080.img', 'WBCT_0081.img', 'WBCT_0082.img', 'WBCT_0083.img', 'WBCT_0084.img', 'WBCT_0085.img', 'WBCT_0086.img', 'WBCT_0087.img', 'WBCT_0088.img', 'WBCT_0089.img', 'WBCT_0090.img', 'WBCT_0091.img', 'WBCT_0092.img', 'WBCT_0093.img', 'WBCT_0094.img', 'WBCT_0095.img', 'WBCT_0096.img', 'WBCT_0097.img', 'WBCT_0098.img', 'WBCT_0099.img', 'WBCT_0100.img', 'WBCT_0101.img', 'WBCT_0102.img', 'WBCT_0103.img', 'WBCT_0104.img', 'WBCT_0105.img', 'WBCT_0106.img', 'WBCT_0107.img', 'WBCT_0108.img', 'WBCT_0109.img', 'WBCT_0110.img', 'WBCT_0111.img', 'WBCT_0112.img', 'WBCT_0113.img', 'WBCT_0114.img', 'WBCT_0115.img', 'WBCT_0116.img', 'WBCT_0117.img', 'WBCT_0118.img', 'WBCT_0119.img', 'WBCT_0120.img', 'WBCT_0121.img', 'WBCT_0122.img', 'WBCT_0123.img', 'WBCT_0124.img', 'WBCT_0125.img', 'WBCT_0126.img', 'WBCT_0127.img', 'WBCT_0128.img', 'WBCT_0129.img', 'WBCT_0130.img', 'WBCT_0131.img', 'WBCT_0132.img', 'WBCT_0133.img', 'WBCT_0134.img', 'WBCT_0135.img', 'WBCT_0136.img', 'WBCT_0137.img', 'WBCT_0138.img', 'WBCT_0139.img', 'WBCT_0140.img', 'WBCT_0141.img', 'WBCT_0142.img', 'WBCT_0143.img', 'WBCT_0144.img', 'WBCT_0145.img', 'WBCT_0146.img', 'WBCT_0147.img', 'WBCT_0148.img', 'WBCT_0149.img', 'WBCT_0150.img', 'WBCT_0151.img', 'WBCT_0152.img', 'WBCT_0153.img', 'WBCT_0154.img', 'WBCT_0155.img', 'WBCT_0156.img', 'WBCT_0157.img', 'WBCT_0158.img', 'WBCT_0159.img', 'WBCT_0160.img', 'WBCT_0161.img', 'WBCT_0162.img', 'WBCT_0163.img', 'WBCT_0164.img', 'WBCT_0165.img', 'WBCT_0166.img', 'WBCT_0167.img', 'WBCT_0168.img', 'WBCT_0169.img', 'WBCT_0170.img', 'WBCT_0171.img', 'WBCT_0172.img', 'WBCT_0173.img', 'WBCT_0174.img', 'WBCT_0175.img', 'WBCT_0176.img', 'WBCT_0177.img', 'WBCT_0178.img', 'WBCT_0179.img', 'WBCT_0180.img', 'WBCT_0181.img', 'WBCT_0182.img', 'WBCT_0183.img', 'WBCT_0184.img', 'WBCT_0185.img', 'WBCT_0186.img', 'WBCT_0187.img', 'WBCT_0188.img', 'WBCT_0189.img', 'WBCT_0190.img', 'WBCT_0191.img', 'WBCT_0192.img', 'WBCT_0193.img', 'WBCT_0194.img', 'WBCT_0195.img', 'WBCT_0196.img', 'WBCT_0197.img', 'WBCT_0198.img', 'WBCT_0199.img', 'WBCT_0200.img', 'WBCT_0201.img', 'WBCT_0202.img', 'WBCT_0203.img', 'WBCT_0204.img', 'WBCT_0205.img', 'WBCT_0206.img', 'WBCT_0207.img', 'WBCT_0208.img', 'WBCT_0209.img', 'WBCT_0210.img', 'WBCT_0211.img', 'WBCT_0212.img', 'WBCT_0213.img', 'WBCT_0214.img', 'WBCT_0215.img', 'WBCT_0216.img', 'WBCT_0217.img', 'WBCT_0218.img', 'WBCT_0219.img', 'WBCT_0220.img', 'WBCT_0221.img', 'WBCT_0222.img', 'WBCT_0223.img', 'WBCT_0224.img', 'WBCT_0225.img', 'WBCT_0226.img', 'WBCT_0227.img', 'WBCT_0228.img', 'WBCT_0229.img', 'WBCT_0230.img', 'WBCT_0231.img', 'WBCT_0232.img', 'WBCT_0233.img', 'WBCT_0234.img', 'WBCT_0235.img', 'WBCT_0236.img', 'WBCT_0237.img', 'WBCT_0238.img', 'WBCT_0239.img', 'WBCT_0240.img', 'WBCT_0241.img', 'WBCT_0242.img', 'WBCT_0243.img', 'WBCT_0244.img', 'WBCT_0245.img', 'WBCT_0246.img', 'WBCT_0247.img', 'WBCT_0248.img', 'WBCT_0249.img', 'WBCT_0250.img', 'WBCT_0251.img', 'WBCT_0252.img', 'WBCT_0253.img', 'WBCT_0254.img', 'WBCT_0255.img', 'WBCT_0256.img', 'WBCT_0257.img', 'WBCT_0258.img', 'WBCT_0259.img', 'WBCT_0260.img', 'WBCT_0261.img', 'WBCT_0262.img', 'WBCT_0263.img', 'WBCT_0264.img', 'WBCT_0265.img', 'WBCT_0266.img', 'WBCT_0267.img', 'WBCT_0268.img', 'WBCT_0269.img', 'WBCT_0270.img', 'WBCT_0271.img', 'WBCT_0272.img', 'WBCT_0273.img', 'WBCT_0274.img', 'WBCT_0275.img', 'WBCT_0276.img', 'WBCT_0277.img', 'WBCT_0278.img', 'WBCT_0279.img', 'WBCT_0280.img', 'WBCT_0281.img', 'WBCT_0282.img', 'WBCT_0283.img', 'WBCT_0284.img', 'WBCT_0285.img', 'WBCT_0286.img', 'WBCT_0287.img', 'WBCT_0288.img', 'WBCT_0289.img', 'WBCT_0290.img', 'WBCT_0291.img', 'WBCT_0292.img', 'WBCT_0293.img', 'WBCT_0294.img', 'WBCT_0295.img', 'WBCT_0296.img', 'WBCT_0297.img', 'WBCT_0298.img', 'WBCT_0299.img', 'WBCT_0300.img', 'WBCT_0301.img', 'WBCT_0302.img', 'WBCT_0303.img', 'WBCT_0304.img', 'WBCT_0305.img', 'WBCT_0306.img', 'WBCT_0307.img', 'WBCT_0308.img', 'WBCT_0309.img', 'WBCT_0310.img', 'WBCT_0311.img', 'WBCT_0312.img', 'WBCT_0313.img', 'WBCT_0314.img', 'WBCT_0315.img', 'WBCT_0316.img', 'WBCT_0317.img', 'WBCT_0318.img', 'WBCT_0319.img', 'WBCT_0320.img', 'WBCT_0321.img', 'WBCT_0322.img', 'WBCT_0323.img', 'WBCT_0324.img', 'WBCT_0325.img', 'WBCT_0326.img', 'WBCT_0327.img', 'WBCT_0328.img', 'WBCT_0329.img', 'WBCT_0330.img', 'WBCT_0331.img', 'WBCT_0332.img', 'WBCT_0333.img', 'WBCT_0334.img', 'WBCT_0335.img', 'WBCT_0336.img', 'WBCT_0337.img', 'WBCT_0338.img', 'WBCT_0339.img', 'WBCT_0340.img', 'WBCT_0341.img', 'WBCT_0342.img', 'WBCT_0343.img', 'WBCT_0344.img', 'WBCT_0345.img', 'WBCT_0346.img', 'WBCT_0347.img', 'WBCT_0348.img', 'WBCT_0349.img', 'WBCT_0350.img']

def train():

    #img path
    img_list = name_list
    print("img_list length :", len(img_list))

    train_ct_path = []
    for i in range(len(img_list)):
        path = TRAIN_DIR_IMG + img_list[i]
        train_ct_path.append(path)

    #img load
    itk_img_0 = sitk.ReadImage(train_ct_path[0])
    train_ct = sitk.GetArrayFromImage(itk_img_0)

    for j in range(1, len(train_ct_path)):
        itk_img = sitk.ReadImage(train_ct_path[j])
        ct_scan = sitk.GetArrayFromImage(itk_img)
        #print(img_list[j], "img shape : ", ct_scan.shape)
        train_ct = np.concatenate((train_ct, ct_scan))

    print("train_ct shape : ", train_ct.shape)
    #concatenate로 모두 합치기
    #(168,512,512), (180,512,512) 이렇게 말고 (500, 512,512) 이렇게

    #expand_dims
    train_ct = np.expand_dims(train_ct, 3)
    print("train_ct_ex shape : ", train_ct.shape)

    #mask path
    mask_list = name_list
    print('mask_list length :', len(mask_list))

    train_mask_path = []
    for k in range(len(mask_list)):
        path = TRAIN_DIR_MASK + mask_list[k]
        train_mask_path.append(path)

    #mask load
    itk_mask_0 = sitk.ReadImage(train_mask_path[0])
    train_mask = sitk.GetArrayFromImage(itk_mask_0)

    for l in range(1, len(train_mask_path)):
        itk_img = sitk.ReadImage(train_mask_path[l])
        mask_scan = sitk.GetArrayFromImage(itk_img)
        #print(mask_list[l], 'mask shape : ', mask_scan.shape)
        train_mask = np.concatenate((train_mask, mask_scan))

    print("train_mask shape : ", train_mask.shape)

    #mask to multi-label
    train_onehot = []
    for m in range(len(train_mask)):
        train_y_onehot_pre = np.zeros(len(np.unique(train_mask)))
        if 0 in train_mask[m]:
            train_y_onehot_pre[0] = 1
        if 1 in train_mask[m]:
            train_y_onehot_pre[1] = 1
        if 2 in train_mask[m]:
            train_y_onehot_pre[2] = 1
        if 3 in train_mask[m]:
            train_y_onehot_pre[3] = 1

        train_onehot.append(train_y_onehot_pre)
    train_onehot = np.array(train_onehot)

    print("train_onehot shape : ", train_onehot.shape)

    #input shape
    (img_rows, img_cols, input_dims) = train_x[0].shape
    input_shape = (img_rows, img_cols, input_dims)
    print(input_shape)

    #hyperparameter
    num_classes = len(train_onehot[0])
    num_ch = 64

    lr = 0.0001
    batch_size = 8
    epochs = 300

    #model fit
    #class없이 그냥 직접짜기

    #input
    img_input = layers.Input(shape = input_shape)

    #block1
    x = layers.Conv2D(num_ch, (7,7), strides = (2,2), padding = 'same', name = 'block1_Conv2D')(img_input)
    x = layers.MaxPooling2D((3,3), strides = (2,2), name = 'block1_maxpoolint')(x)
    x = layers.BatchNormalization()(x)
    input_layer = layers.Activation('relu')(x)

    #block2
    x = layers.Conv2D(num_ch, (3,3), padding = 'same', name = "block2_Conv2D_1")(input_layer)
    x = layers.Conv2D(num_ch, (3,3), padding = 'same', name = "blaock2_Conv2D_2")(x)
    x = layers.add([x, input_layer])
    x = layers.BatchNormalization()(x)
    input_layer = layers.Activation('relu')(x)

    #block3
    x = layers.Conv2D(num_ch, (3,3), padding = 'same', name = "block3_Conv2D_1")(input_layer)
    x = layers.Conv2D(num_ch, (3,3), padding = 'same', name = "block3_Conv2D_2")(x)
    x = layers.add([x, input_layer])
    x = layers.BatchNormalization()(x)
    input_layer = layers.Activation('relu')(x)

    #block4
    x = layers.Conv2D(num_ch*2, (3,3), padding = 'same', name = "block4_Conv2D_1")(input_layer)
    shortcut = layers.MaxPooling2D((2,2), name = "block4_maxpooling")(x)
    x = layers.Conv2D(num_ch*2, (3,3), padding = 'same', name = "block4_Conv2D_2")(shortcut)
    x = layers.add([x, shortcut])
    x = layers.BatchNormalization()(x)
    input_layer = layers.Activation('relu')(x)

    #block5
    x = layers.Conv2D(num_ch*2, (3,3), padding = 'same', name = "block5_Conv2D_1")(input_layer)
    x = layers.Conv2D(num_ch*2, (3,3), padding = 'same', name = "block5_Conv2D_2")(x)
    x = layers.add([x, input_layer])
    x = layers.BatchNormalization()(x)
    input_layer = layers.Activation('relu')(x)

    #block6
    x = layers.Conv2D(num_ch*4, (3,3), padding = 'same', name = "block6_Conv2D_1")(input_layer)
    shortcut = layers.MaxPooling2D((2,2), name = "block6_maxpooling")(x)
    x = layers.Conv2D(num_ch*4, (3,3), padding = 'same', name = "block4_Conv2D_2")(shortcut)
    x = layers.add([x, shortcut])
    x = layers.BatchNormalization()(x)
    input_layer = layers.Activation('relu')(x)

    #block7
    x = layers.Conv2D(num_ch*4, (3,3), padding = 'same', name = "block7_Conv2D_1")(input_layer)
    x = layers.Conv2D(num_ch*4, (3,3), padding = 'same', name = "block7_Conv2D_2")(x)
    x = layers.add([x, input_layer])
    x = layers.BatchNormalization()(x)
    input_layer = layers.Activation('relu')(x)

    #flatten
    x = layers.Flatten(name = "flatten")(input_layer)
    x = layers.Dense(1024, name = "flatten_1")(x)
    x = layers.Dense(num_classes, activation = 'sigmoid', name = "prediction")(x)

    #model
    model = models.Model(img_input, x)

    #optimizer
    adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    #compile
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])

    #learning
    history = model.fit(train_ct, train_onehot, validation_split = 0.2, epochs = epochs, batch_size = batch_size)


if __name__ == "__main__":
  train()
