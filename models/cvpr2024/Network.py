import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torchvision.models.resnet import resnet18
from models.resnet20_cifar import resnet20


from torchvision import transforms

class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args

        if self.args.dataset in ['cifar100']:
            self.encoder = resnet20()
            self.feature_dim = 64

            self.fc1 = nn.Linear(self.feature_dim, self.args.num_classes, bias=False)

            self.num_features = 640
            
        if self.args.dataset in ['mini_imagenet']:
            #self.encoder = resnet12()
            #self.feature_dim = 640

            self.encoder = resnet18(False, args)
            self.encoder.fc = nn.Identity()
            self.feature_dim = 512

            self.fc1 = nn.Linear(self.feature_dim, self.args.num_classes, bias=False)

        elif self.args.dataset == 'cub200':
            self.encoder = resnet18(True, args)  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            self.feature_dim = 512
            self.num_features = 512
            self.encoder.fc = nn.Identity()

            self.fc1 = nn.Linear(self.feature_dim, self.args.num_classes, bias=False)

        self.session = 0
        self.test = False
    
    def forward_metric(self, x, pos=None):
        inp = x

        x = self.encoder(x)
        fc = self.fc1.weight

        x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))
        x = self.args.temperature * x

        return x

    def encode(self, x):
        x = self.encoder(x) # (b, c, h, w)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x

    def forward(self, input, pos=None):
        if self.mode != 'encoder':
            input = self.forward_metric(input,pos)
            return input
        elif self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            raise ValueError('Unknown mode')

    def update_fc(self,dataloader,class_list,session):
        feats = []
        labels = []
        with torch.no_grad():
            if self.args.dataset == 'cifar100':
                size = 32
                dataloader.dataset.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762))])
            elif self.args.dataset == 'mini_imagenet':
                size = 84
                dataloader.dataset.transform = transforms.Compose([
                    transforms.Resize([96, 96]),
                    transforms.CenterCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
            else:
                size = 224
                dataloader.dataset.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

            for batch in dataloader:
                data, label = [_.cuda() for _ in batch]
                feats.append(self.encoder(data).detach())

                labels.append(label)

        feats = torch.cat(feats, dim=0)
        labels = torch.cat(labels,dim=0)

        for ii in range(labels.unique().shape[0]):
            self.fc1.weight.data[labels.min()+ii] = feats[labels==labels.min()+ii].mean(dim=0)
