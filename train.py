import argparse
import os
import shutil

import torch
import torch.nn as nn
from conf import global_settings as settings
import torch.optim as optim

import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import time


from utils import  get_training_dataloader, get_valid_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from utils import get_network

def train(epoch):

    start = time.time()
    #模型切换到训练态更新参数
    net.train()
    for batch_index, (data, labels) in enumerate(training_loader):

        if settings.GPU:
            labels = labels.cuda()
            #data = data.cuda()
            data = {key: value.cuda() for (key, value) in data.items()}

        optimizer.zero_grad()

        outputs = net(data).view(-1)
        outputs=outputs.to(torch.float)
        labels=labels.to(torch.float)

        #
        print(outputs)
        print()
        print(labels)

        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(training_loader) + batch_index + 1

        # for name, para in last_layer.named_parameters():
        #     if 'weight' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
        #     if 'bias' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * settings.BATCH_SIZE + len(data),
            total_samples=len(training_loader.dataset)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

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
            #data = data.cuda()
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

    # add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(valid_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(valid_loader.dataset), epoch)

    return correct.float() / len(valid_loader.dataset)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    args = parser.parse_args()
    net = get_network(args.net)

    if settings.GPU:
        net = net.cuda()
    # data preprocessing:
    training_loader = get_training_dataloader(
        num_workers=0,
        batch_size=settings.BATCH_SIZE,
        shuffle=True
    )

    valid_loader = get_valid_dataloader(
        num_workers=0,
        batch_size=settings.BATCH_SIZE,
        shuffle=True
    )

    #loss_function = nn.CrossEntropyLoss()
    loss_function = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=settings.LR, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES,
                                                     gamma=0.2)  # learning rate decay
    iter_per_epoch = len(training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * settings.WARMUP)
    if settings.RESUME:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net),
                                           fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)
    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    # use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    # since tensorboard can't overwrite old values
    # so the only way is to create a new tensorboard log
    #TensorBoard 的使用
    #
    writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, settings.NET, settings.TIME_NOW))
    # input_tensor = torch.Tensor(1, 1, 32, 32)
    # 可视化模型计算图
    # if settings.GPU:
    #     input_tensor = input_tensor.cuda()
    # writer.add_graph(net, input_tensor)

    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    pth_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    #中断恢复
    if settings.RESUME:
        #加载目前最优参数并输出结果
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))
        #加载目前最近批次参数继续训练
        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))
        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
    for epoch in range(1, settings.EPOCH + 1):
        if epoch > settings.WARMUP:
            train_scheduler.step(epoch)
        #恢复到模型训练的批次
        if settings.RESUME:
            if epoch <= resume_epoch:
                continue
        train(epoch)
        acc = eval_training(epoch)
        # start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = pth_path.format(net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = pth_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
    best_result = os.path.join(checkpoint_path, best_acc_weights(checkpoint_path))
    best_result_path=r'./result'
    if not os.path.exists(best_result_path):
        os.makedirs(best_result_path)
    shutil.copyfile(best_result, os.path.join(best_result_path,args.net+'-'+settings.TIME_NOW+'.pth'))
    writer.close()



