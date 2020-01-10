pip install SimpleITK

import SimpleITK as sitk
import numpy as np

def sitk_load(path):

    # .img .hdr 이미지 읽어오기
    itk_img = sitk.ReadImage(path)

    #x,y,z축을 고려하여 ct이미지로 불러오고 형태는 np array
    ct_sacn = sitk.GetArrayFromImage(itk_img)

    origin = np.array(list(reversed(itkimage.GetOrigin())))

    #spacing
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing
    
