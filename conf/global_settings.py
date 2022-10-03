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
DATASET_PATH=r'./data/dataset1'
TRAIN_DATASET_PATH=os.path.join(DATASET_PATH,'train')
VALID_DATASET_PATH=os.path.join(DATASET_PATH,'val')
TEST_DATASET_PATH=os.path.join(DATASET_PATH,'test')

#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

#total training epoches
EPOCH = 200
#batch size for dataloader
BATCH_SIZE=16
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








