'''
预处理工具类
'''
import SimpleITK as sitk
import numpy as np
import os

def normalize(img):
    Min=np.min(img)
    Max=np.max(img)
    img_new=(img-Min)/(Max-Min)
    return img_new

def zscore(img):
    mean=np.mean(img)
    std=np.std(img)
    return (img-mean)/std

# def getnonzeros(data:np.ndarray):
#     nonIndex=[[],[],[]]
#     for i in range(0,3):
#         for j in range(0,data.shape[i]):
#             if i==0:
#                 temp=np.sum(data[j,:,:]).squeeze()
#             elif i==1:
#                 temp=np.sum(data[:,j,:]).squeeze()
#             elif i==2:
#                 temp=np.sum(data[:,:,j]).squeeze()
#             if temp !=0:
#                 nonIndex[i].append(j)
#     return nonIndex
def getnonzeros(data:np.ndarray,Tran=False):
    if Tran:
        data=data.transpose((2,1,0))
    nonIndex=[[],[],[]]
    for i in range(0,3):
        for j in range(0,data.shape[i]):
            if i==0:
                temp=np.sum(data[j,:,:]).squeeze()
            elif i==1:
                temp=np.sum(data[:,j,:]).squeeze()
            elif i==2:
                temp=np.sum(data[:,:,j]).squeeze()
            if temp !=0:
                nonIndex[i].append(j)
    return nonIndex
def createdirs(*dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)


def setWindow(data):
    wincenter = -800
    winwidth = 2000
    min = int(wincenter - winwidth / 2.0)
    max = int(wincenter + winwidth / 2.0)

    intensityWindow = sitk.IntensityWindowingImageFilter()
    intensityWindow.SetWindowMaximum(max)
    intensityWindow.SetWindowMinimum(min)
    new_data = intensityWindow.Execute(data)
    return new_data

"""
统一Size
X轴和Y轴的Size和Spacing没有变化，
Z轴的Size和Spacing有变化
"""
def resample2Size(img,dst_size=[32,32,32]):
    original_size=img.GetSize()
    original_spacing = img.GetSpacing()
    new_space =[old_size*old_space/new_size for old_size,old_space,new_size in zip(original_size,original_spacing,dst_size)]
    resampled_img=sitk.Resample(img,dst_size,sitk.Transform(),sitk.sitkLinear,img.GetOrigin(),new_space,img.GetDirection(),0,img.GetPixelID())
    return resampled_img