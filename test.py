import argparse
import os
import torch
import torch.nn as nn
from conf import global_settings as settings
import torch.optim as optim

import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import time
from datetime import datetime

#画ROC曲线
from sklearn.metrics import roc_curve,auc,confusion_matrix,classification_report
import matplotlib.pyplot as plt

#导入模型
from models.resnet import resnet18

from utils import get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights, get_network

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    args = parser.parse_args()
    net = get_network(args.net,useCli=True)
    if settings.GPU:
        net=net.cuda()

    cifar100_test_loader = get_test_dataloader(
        #settings.CIFAR100_PATH,
        num_workers=0,
        batch_size=settings.BATCH_SIZE,
    )

    net.load_state_dict(torch.load(r'checkpoint\4\cresnet18-1.pth'))
    print(net)
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0
    y_true=[]
    y_score=[]
    y_pred=[]

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(cifar100_test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))

            if settings.GPU:
                #image = image.cuda()
                image = {key: value.cuda() for (key, value) in image.items()}
                label = label.cuda()
                # print('GPU INFO.....')
                # print(torch.cuda.memory_summary(), end='')

            output = net(image)
            #获取最大可能的切片
            # softmax=nn.LogSoftmax(dim=1)
            # output=softmax(output)

            '''
                第一个是多分类使用
                第二个是二分类使用
            '''
            # pro, pred = output.topk(1, 1, largest=True, sorted=True)
            # label = label.view(label.size(0), -1).expand_as(pred)
            output=output.view(-1)
            print(output)
            print(label)
            pred = (output > 0.5).to(torch.int)

            correct = pred.eq(label).float()

            # 记录label和概率用于画ROC曲线
            y_true.extend(label.cpu().numpy().reshape(-1).tolist())
            y_score.extend(output.cpu().numpy().reshape(-1).tolist())
            y_pred.extend(pred.cpu().numpy().reshape(-1).tolist())

            correct_1 += correct.sum()

    # if settings.GPU:
    #     print('GPU INFO.....')
    #     print(torch.cuda.memory_summary(), end='')
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
    P,N=TP+FN,FP+TN
    accuracy=(TP+TN)/(P+N)
    sensitive=TP/P
    specificity=TN/N
    precision=TP/(TP+FP)
    print(f'accuracy:{accuracy} ,sensitive:{sensitive} ,specificity:{specificity} ,precision:{precision}')
    print(classification_report(y_true,y_pred))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))

    #画ROC曲线
    # print(y_true)
    # print(y_score)
    fpr, tpr, thersholds = roc_curve(y_true, y_score)
    auc_value=auc(fpr,tpr)
    # 逆序输出标签，概率
    # for i, value in enumerate(thersholds):
    #     print("%f %f %f" % (fpr[i], tpr[i], value))
    roc_auc = auc(fpr, tpr)
    print(f'auc={auc_value}')
    fig=plt.figure()
    plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)

    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    fig.savefig(r'.\ROC curve.png')
