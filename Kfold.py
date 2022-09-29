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

from models.resnet import resnet18

from utils import  get_kfolder_dataloader,WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from utils import get_network
import shutil

def train(epoch):

    start = time.time()
    #模型切换到训练态更新参数
    net.train()
    for batch_index, (data, labels) in enumerate(training_loader):

        if settings.GPU:
            labels = labels.cuda()
            # images = images.cuda()
            data = {key: value.cuda() for (key, value) in data.items()}

        optimizer.zero_grad()

        outputs = net(data).view(-1)
        outputs=outputs.to(torch.float)
        labels=labels.to(torch.float)

        #
        # print(outputs)
        # print()
        # print(labels)

        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        # for name, para in last_layer.named_parameters():
        #     if 'weight' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
        #     if 'bias' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * settings.BATCH_SIZE + len(data['image']),
            total_samples=len(training_loader.dataset)
        ))

        #update training loss for each iteration
        # writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= settings.WARMUP:
            warmup_scheduler.step()

    # for name, param in net.named_parameters():
    #     layer, attr = os.path.splitext(name)
    #     attr = attr[1:]
    #     writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (data, labels) in valid_loader:

        if settings.GPU:
            #images = images.cuda()
            data={key:value.cuda() for (key,value) in data.items()}
            labels = labels.cuda()

        outputs = net(data)
        outputs = net(data).view(-1)
        outputs = outputs.to(torch.float)
        labels = labels.to(torch.float)

        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        """   第一种的是输出为多类别的情况下用来判断识别类的方法
                第二种为二分类情况下输出为概率使用的方法
         """
        #_, preds = outputs.max(1)
        preds=(outputs>0.5).to(torch.int)
        correct += preds.eq(labels).sum()

    finish = time.time()
    if settings.GPU:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(valid_loader.dataset),
        correct.float() / len(valid_loader.dataset),
        finish - start
    ))
    print()

    # add informations to tensorboard
    # if tb:
    #     writer.add_scalar('Test/Average loss', test_loss / len(valid_loader.dataset), epoch)
    #     writer.add_scalar('Test/Accuracy', correct.float() / len(valid_loader.dataset), epoch)

    return correct.float() / len(valid_loader.dataset)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    #默认值为9，即train+valid：test=9:1,对train+valid按9折划分，最终train：val：test=8:1:1
    parser.add_argument('-k',type=int,help='the folder of validation',default=9)
    args = parser.parse_args()
    net = get_network(args.net)
    k=args.k
    if settings.GPU:
        net = net.cuda()
    checkpoint_path=os.path.join(settings.CHECKPOINT_PATH, args.net,'Kfold')
    best_result_path=os.path.join(checkpoint_path,'result')
    if not os.path.exists(best_result_path):
        os.makedirs(best_result_path)
    # data preprocessing:
    for i,(training_loader,valid_loader) in enumerate(get_kfolder_dataloader(k)):
    
    
        #loss_function = nn.CrossEntropyLoss()
        loss_function = nn.BCELoss()
        optimizer = optim.SGD(net.parameters(), lr=settings.LR, momentum=0.9, weight_decay=5e-4)
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES,
                                                         gamma=0.2)  # learning rate decay
        iter_per_epoch = len(training_loader)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * settings.WARMUP)

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
        # create checkpoint folder to save model
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')
    
        best_acc = 0.0
        for epoch in range(1, settings.EPOCH + 1):
            if epoch > settings.WARMUP:
                train_scheduler.step(epoch)
            #恢复到模型训练的批次
            train(epoch)
            acc = eval_training(epoch)
            # start to save best performance model after learning rate decay to 0.01
            if epoch > settings.MILESTONES[1] and best_acc < acc:
                weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
                print('saving weights file to {}'.format(weights_path))
                torch.save(net.state_dict(), weights_path)
                best_acc = acc
                continue
    
            if not epoch % settings.SAVE_EPOCH:
                weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
                print('saving weights file to {}'.format(weights_path))
                torch.save(net.state_dict(), weights_path)
        best_result=os.join.path(checkpoint_path,best_acc_weights(checkpoint_path))
        shutil.copyfile(best_result,os.join.path(best_result_path,f'{i}.pth'))
        #writer.close()



