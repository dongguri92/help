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

TEST_DIR = '/data/test/'

name_list = ['WBCT_0701.img', 'WBCT_0702.img', 'WBCT_0703.img', 'WBCT_0704.img', 'WBCT_0705.img', 'WBCT_0706.img', 'WBCT_0707.img', 'WBCT_0708.img', 'WBCT_0709.img', 'WBCT_0710.img', 'WBCT_0711.img', 'WBCT_0712.img', 'WBCT_0713.img', 'WBCT_0714.img', 'WBCT_0715.img', 'WBCT_0716.img', 'WBCT_0717.img', 'WBCT_0718.img', 'WBCT_0719.img', 'WBCT_0720.img', 'WBCT_0721.img', 'WBCT_0722.img', 'WBCT_0723.img',
 'WBCT_0724.img', 'WBCT_0725.img', 'WBCT_0726.img', 'WBCT_0727.img', 'WBCT_0728.img', 'WBCT_0729.img', 'WBCT_0730.img', 'WBCT_0731.img', 'WBCT_0732.img', 'WBCT_0733.img', 'WBCT_0734.img', 'WBCT_0735.img',
 'WBCT_0736.img', 'WBCT_0737.img', 'WBCT_0738.img', 'WBCT_0739.img', 'WBCT_0740.img', 'WBCT_0741.img', 'WBCT_0742.img', 'WBCT_0743.img', 'WBCT_0744.img', 'WBCT_0745.img', 'WBCT_0746.img', 'WBCT_0747.img',
 'WBCT_0748.img', 'WBCT_0749.img', 'WBCT_0750.img', 'WBCT_0751.img', 'WBCT_0752.img', 'WBCT_0753.img', 'WBCT_0754.img', 'WBCT_0755.img', 'WBCT_0756.img', 'WBCT_0757.img', 'WBCT_0758.img', 'WBCT_0759.img',
 'WBCT_0760.img', 'WBCT_0761.img', 'WBCT_0762.img', 'WBCT_0763.img', 'WBCT_0764.img', 'WBCT_0765.img']

def inference():
    #model load
    model = models.load_model('/data/volume/logs/model_20200115_1.h5')

    #img path
    img_list = name_list
    print("img_list length :", len(img_list))

    test_ct_path = []
    for i in range(len(img_list)):
        path = TEST_DIR + img_list[i]
        test_ct_path.append(path)

    print(test_ct_path)


    #img load
    itk_img_0 = sitk.ReadImage(test_ct_path[0])
    test_ct = sitk.GetArrayFromImage(itk_img_0)

    print(test_ct.shape)

    for j in range(1, len(test_ct_path)):
        itk_img = sitk.ReadImage(test_ct_path[j])
        ct_scan = sitk.GetArrayFromImage(itk_img)
        train_ct = np.concatenate((test_ct, ct_scan))

    print("test_ct shape : ", test_ct.shape)


    #expand_dims
    test_ct = np.expand_dims(test_ct, 3)
    print("test_ct_ex shape : ", test_ct.shape)

    #model 저장되어있다고 가정
    model.predict(test_ct)


if __name__ == "__main__":
    inference()
