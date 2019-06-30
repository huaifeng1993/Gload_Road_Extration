# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# Written by SHEN HUIXIANG  (shhuixi@qq.com)
# Created On: 2018-12-01
# --------------------------------------------------------
from model.dfanet import xceptionAx3
from model import UNet
from road_extration import *
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import datetime
from torch.optim import lr_scheduler
from config import Config
from train import Trainer
from lr_scheduler import WarmAndReduce_LR

#from loss import WeightedBCELoss2d
#from dice_loss import 
from dice_bce_loss import dice_bce_loss
#from torch.nn import BCELoss
from loss import lovasz_hinge
cfig = Config()
net =net = UNet(n_channels=3, n_classes=1)
#criterion =lovasz_hinge  
criterion = dice_bce_loss()


optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)  #select the optimizer
exp_lr_scheduler=WarmAndReduce_LR(optimizer,0.01,400,use_warmup=True)
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.1)
# create the train_dataset_loader and val_dataset_loader.

data_train=RoadExtrationDataset(root_dir="./data/RoadExtraction/train",
                             n_train=5000,
                             transform=transforms.Compose([Rescale(512),
                                                           RandomFlip(),
                                                           RandomScale((0.75,1.2)),
                                                           RandomCorp(256),
                                                           Normalize([0.410,0.383,0.288],[0.156,0.126,0.123]),
                                                           ToTensor(),
                                                           ],))

data_val=RoadExtrationDataset(root_dir="./data/RoadExtraction/train",
                             n_train=5000,
                             val=True,
                             transform=transforms.Compose([Rescale(512),
                                                           Normalize([0.410,0.383,0.288],[0.156,0.126,0.123]),
                                                           ToTensor(),
                                                           ],))


train_dataloader = DataLoader(
    data_train, batch_size=16, shuffle=True, num_workers=4)

val_dataloader = DataLoader(
    data_val, batch_size=4, shuffle=False, num_workers=4)

trainer = Trainer('training', optimizer,exp_lr_scheduler, net, cfig, './log')
#trainer.load_weights(trainer.find_last()) 
trainer.train(train_dataloader, val_dataloader, criterion, 400)
#trainer.evaluate(val_dataloader)
print('Finished Training')
