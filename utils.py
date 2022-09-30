""" helper function

author baiyu
"""
import os
import sys
import re
import datetime
from PIL import Image
import numpy as np

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from torchvision.datasets import DatasetFolder
from conf import global_settings as gs
from sklearn.model_selection import KFold
import re



def get_network(net):
    """ return given network
    """
    res = re.search('([a-z]+)([1-9]+)', net)
    model_name=res.group(1)
    depth=int(res.group(2))
    if model_name == 'resnet':
        from models import resnet
        return resnet.generate_model(depth)
    elif model_name=='densenet':
        from models import densenet
        return densenet.generate_model(depth)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    return net

class ImageClinicalDataset(DatasetFolder):
    '''
        加载图像和临床数据的numpy数组
    '''
    def __init__(self,root,loader,extensions=None,transform=None,target_transform=None,is_valid_file=None):
        DatasetFolder.__init__(self,root,loader,extensions,transform,target_transform,is_valid_file)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample['image'] = self.transform(sample['image'])
            
        if self.target_transform is not None:
            target = self.target_transform(target)
        #image 已经在transform中转换为了Tensor，这里仅仅改变类型为Float
        sample['image']=sample['image'].float()
        sample['image']=sample['image'].unsqueeze(0)
        sample['cdata']=torch.from_numpy(sample['cdata']).float()
        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

def loader(x):
    data=np.load(x)

    return {key:data[key] for key in data.files}

def get_training_dataloader(batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomRotation(15),
        #transforms.Resize((32,32)),
        transforms.ToTensor(),
        #transforms.Normalize(mean, std)
    ])


    #train_set=DatasetFolder(gs.TRAIN_DATASET_PATH,loader=lambda x:np.load(x),extensions='npz',transform=transform_train)

    #使用自定义数据加载器加载npz文件
    train_set = ImageClinicalDataset(gs.TRAIN_DATASET_PATH, loader=loader, extensions='npz',
                              transform=transform_train)
    train_loader=DataLoader(train_set,shuffle=shuffle,num_workers=num_workers,batch_size=batch_size)
    return train_loader
def get_valid_dataloader(batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_valid = transforms.Compose([
        #transforms.ToPILImage(),
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomRotation(15),
        #transforms.Resize((32,32)),
        transforms.ToTensor(),
        #transforms.Normalize(mean, std)
    ])

    #valid_set=DatasetFolder(gs.VALID_DATASET_PATH,loader=lambda x:np.load(x),extensions='npz',transform=transform_valid)

    #使用自定义数据加载器加载npz文件
    valid_set = ImageClinicalDataset(gs.VALID_DATASET_PATH, loader=loader, extensions='npz',
                              transform=transform_valid)
    valid_loader=DataLoader(valid_set,shuffle=shuffle,num_workers=num_workers,batch_size=batch_size)
    return valid_loader
def get_test_dataloader(batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        #transforms.Resize((32,32)),
        transforms.ToTensor(),
        #transforms.Normalize(mean, std)
    ])

    #test_set = DatasetFolder(gs.TEST_DATASET_PATH, loader=lambda x: np.load(x), extensions='npz',
    #                            transform=transform_test)
    # #使用自定义数据加载器
    test_set = ImageClinicalDataset(gs.TEST_DATASET_PATH, loader=loader, extensions='npz',
                                                        transform=transform_test)
    test_loader = DataLoader(test_set, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return test_loader

def get_kfolder_dataloader(k,batch_size=gs.BATCH_SIZE, num_workers=0, shuffle=True):
    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ])

    kf=KFold(n_splits=k,shuffle=True,random_state=0)
    data=ImageClinicalDataset(gs.TRAIN_DATASET_PATH, loader=loader, extensions='npz',
                                                        transform=transform_test)
    for train_index,val_index in kf.split(data):
        train_fold=torch.utils.data.dataset.Subset(data, train_index)
        val_fold = torch.utils.data.dataset.Subset(data, val_index)
        train_loader = DataLoader(train_fold, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        val_loader = DataLoader(val_fold, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        yield (train_loader,val_loader)


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]