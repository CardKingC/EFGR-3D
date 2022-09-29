""" configurations for this project

author baiyu
"""
import os
from datetime import datetime

'''
    数据集位置，
    1、单独训练时(train.py)，须有train，val，test目录
    2、使用交叉验证时，须有train，test目录，val目录可以为空或没有
'''
DATASET_PATH=r'./data/dataset5_last'
TRAIN_DATASET_PATH=os.path.join(DATASET_PATH,'train')
VALID_DATASET_PATH=os.path.join(DATASET_PATH,'val')
TEST_DATASET_PATH=os.path.join(DATASET_PATH,'test')

#mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

#CIFAR100_TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
#CIFAR100_TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)

#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

#total training epoches
EPOCH = 200
#batch size for dataloader
BATCH_SIZE=128
#Learning rate
LR=0.1
# warm up the training phase
WARMUP=1
#learning rate 调整的批次
MILESTONES = [60, 120, 160]
# use gpu or not
GPU=True
#resume training 恢复中断的训练
RESUME=False
# net type
NET='resnet18'
# input channels
INPUT_CHANNELS=1

#initial learning rate
#INIT_LR = 0.1

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10








