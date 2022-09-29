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
def resampleSize(sitkImage, outSize):
    #重采样函数
    euler3d = sitk.Euler3DTransform()

    xsize, ysize, zsize = sitkImage.GetSize()
    xspacing, yspacing, zspacing = sitkImage.GetSpacing()
    new_spacing_x = zspacing/(outSize[0]/float(xsize))
    new_spacing_y = zspacing/(outSize[1]/float(ysize))
    new_spacing_z = zspacing/(outSize[2]/float(zsize))

    origin = sitkImage.GetOrigin()
    direction = sitkImage.GetDirection()
    #根据新的spacing 计算新的size
    newspace = (new_spacing_x, new_spacing_y, new_spacing_z)
    sitkImage = sitk.Resample(sitkImage,outSize,euler3d,sitk.sitkNearestNeighbor,origin,newspace,direction)
    return sitkImage