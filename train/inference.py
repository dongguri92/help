import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import SimpleITK as sitk
import os
from time import sleep
import csv

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import models

TEST_DIR = '/data/test/'
OUT_DIR = '/data/output/'

name_list = ['WBCT_0701.img', 'WBCT_0702.img', 'WBCT_0703.img', 'WBCT_0704.img', 'WBCT_0705.img', 'WBCT_0706.img', 'WBCT_0707.img', 'WBCT_0708.img', 'WBCT_0709.img', 'WBCT_0710.img', 'WBCT_0711.img', 'WBCT_0712.img', 'WBCT_0713.img', 'WBCT_0714.img', 'WBCT_0715.img', 'WBCT_0716.img', 'WBCT_0717.img', 'WBCT_0718.img', 'WBCT_0719.img', 'WBCT_0720.img', 'WBCT_0721.img', 'WBCT_0722.img', 'WBCT_0723.img',
 'WBCT_0724.img', 'WBCT_0725.img', 'WBCT_0726.img', 'WBCT_0727.img', 'WBCT_0728.img', 'WBCT_0729.img', 'WBCT_0730.img', 'WBCT_0731.img', 'WBCT_0732.img', 'WBCT_0733.img', 'WBCT_0734.img', 'WBCT_0735.img',
 'WBCT_0736.img', 'WBCT_0737.img', 'WBCT_0738.img', 'WBCT_0739.img', 'WBCT_0740.img', 'WBCT_0741.img', 'WBCT_0742.img', 'WBCT_0743.img', 'WBCT_0744.img', 'WBCT_0745.img', 'WBCT_0746.img', 'WBCT_0747.img',
 'WBCT_0748.img', 'WBCT_0749.img', 'WBCT_0750.img', 'WBCT_0751.img', 'WBCT_0752.img', 'WBCT_0753.img', 'WBCT_0754.img', 'WBCT_0755.img', 'WBCT_0756.img', 'WBCT_0757.img', 'WBCT_0758.img', 'WBCT_0759.img',
 'WBCT_0760.img', 'WBCT_0761.img', 'WBCT_0762.img', 'WBCT_0763.img', 'WBCT_0764.img', 'WBCT_0765.img']

csv_list = ['WBCT_0701.csv', 'WBCT_0702.csv', 'WBCT_0703.csv', 'WBCT_0704.csv', 'WBCT_0705.csv', 'WBCT_0706.csv', 'WBCT_0707.csv', 'WBCT_0708.csv', 'WBCT_0709.csv', 'WBCT_0710.csv', 'WBCT_0711.csv',
 'WBCT_0712.csv', 'WBCT_0713.csv', 'WBCT_0714.csv', 'WBCT_0715.csv', 'WBCT_0716.csv', 'WBCT_0717.csv', 'WBCT_0718.csv', 'WBCT_0719.csv', 'WBCT_0720.csv', 'WBCT_0721.csv', 'WBCT_0722.csv', 'WBCT_0723.csv',
 'WBCT_0724.csv', 'WBCT_0725.csv', 'WBCT_0726.csv', 'WBCT_0727.csv', 'WBCT_0728.csv', 'WBCT_0729.csv', 'WBCT_0730.csv', 'WBCT_0731.csv', 'WBCT_0732.csv', 'WBCT_0733.csv', 'WBCT_0734.csv', 'WBCT_0735.csv',
 'WBCT_0736.csv', 'WBCT_0737.csv', 'WBCT_0738.csv', 'WBCT_0739.csv', 'WBCT_0740.csv', 'WBCT_0741.csv', 'WBCT_0742.csv', 'WBCT_0743.csv', 'WBCT_0744.csv', 'WBCT_0745.csv', 'WBCT_0746.csv', 'WBCT_0747.csv',
 'WBCT_0748.csv', 'WBCT_0749.csv', 'WBCT_0750.csv', 'WBCT_0751.csv', 'WBCT_0752.csv', 'WBCT_0753.csv', 'WBCT_0754.csv', 'WBCT_0755.csv', 'WBCT_0756.csv', 'WBCT_0757.csv', 'WBCT_0758.csv', 'WBCT_0759.csv',
 'WBCT_0760.csv', 'WBCT_0761.csv', 'WBCT_0762.csv', 'WBCT_0763.csv', 'WBCT_0764.csv', 'WBCT_0765.csv']

name = ['WBCT_0701', 'WBCT_0702', 'WBCT_0703', 'WBCT_0704', 'WBCT_0705', 'WBCT_0706', 'WBCT_0707', 'WBCT_0708', 'WBCT_0709', 'WBCT_0710', 'WBCT_0711', 'WBCT_0712', 'WBCT_0713', 'WBCT_0714', 'WBCT_0715', 'WBCT_0716', 'WBCT_0717', 'WBCT_0718','WBCT_0719', 'WBCT_0720', 'WBCT_0721', 'WBCT_0722', 'WBCT_0723', 'WBCT_0724', 'WBCT_0725', 'WBCT_0726', 'WBCT_0727', 'WBCT_0728', 'WBCT_0729', 'WBCT_0730', 'WBCT_0731', 'WBCT_0732', 'WBCT_0733', 'WBCT_0734', 'WBCT_0735', 'WBCT_0736', 'WBCT_0737', 'WBCT_0738', 'WBCT_0739', 'WBCT_0740', 'WBCT_0741', 'WBCT_0742', 'WBCT_0743', 'WBCT_0744', 'WBCT_0745', 'WBCT_0746', 'WBCT_0747', 'WBCT_0748', 'WBCT_0749', 'WBCT_0750', 'WBCT_0751', 'WBCT_0752', 'WBCT_0753', 'WBCT_0754', 'WBCT_0755', 'WBCT_0756', 'WBCT_0757', 'WBCT_0758', 'WBCT_0759', 'WBCT_0760', 'WBCT_0761', 'WBCT_0762', 'WBCT_0763', 'WBCT_0764', 'WBCT_0765']

def inference():
    #csv 형식 --> 파일자체가 없나보다

    #model load
    model = models.load_model('/data/volume/logs/model_20200115_6.h5')

    #img path
    img_list = name_list
    print("img_list length :", len(img_list))

    test_ct_path = []
    for i in range(len(img_list)):
        path = TEST_DIR + img_list[i]
        test_ct_path.append(path)

    print(test_ct_path)

    #per patint
    for j in range(len(test_ct_path)):
        print(img_list[j])
        itk_img = sitk.ReadImage(test_ct_path[j])
        test_ct = sitk.GetArrayFromImage(itk_img)
        test_ct = np.expand_dims(test_ct, 3)
        predictions = model.predict(test_ct)

        list_1 = []
        list_2 = []
        list_3 = []
        for k in range(len(predictions)):
            list_1.append(predictions[k][1])
            list_2.append(predictions[k][2])
            list_3.append(predictions[k][3])

        lesion_1 = float(np.max(list_1))
        lesion_2 = float(np.max(list_2))
        lesion_3 = float(np.max(list_3))

        '''
        f = open(OUT_DIR + csv_list[j], 'w', encoding='utf-8', newline='')
        wr = csv.writer(f)

        wr.writerow(['ID', name[j], ''])
        wr.writerow(['Hemothorax', lesion_1])
        wr.writerow(['Pneumothorax', lesion_2])
        wr.writerow(['Hemoperitoneum', lesion_3])
        wr.writerow(',,')
        for w in range(len(predictions)):
            wr.writerow([float(predictions[w][1]), float(predictions[w][2]), float(predictions[w][3])])
        f.close()
        '''

        #pandas 쓰기
        my_df1 = pd.DataFrame(data=['Hemothorax','Pneumothorax','Hemoperitoneum',np.nan], index=range(0,4), columns=['ID'])
        my_df2 = pd.DataFrame(data=[lesion_1, lesion_2, lesion_3, np.nan], index=range(0,4), columns=[name[j]])
        my_df3 = pd.DataFrame(data=[np.nan,np.nan,np.nan,np.nan], index=range(0,4), columns=[''])
        df_pre = my_df1.join(my_df2, how='left')
        df = df_pre.join(my_df3, how = 'left')
        for w in range(len(predictions)):
            df.loc[w+4] = [predictions[w][1], predictions[w][2], predictions[w][3]]
        print(OUT_DIR + csv_list[j])
        print(df)
        df.to_csv(OUT_DIR + csv_list[j], index = False)

if __name__ == "__main__":
    inference()
    #pass
