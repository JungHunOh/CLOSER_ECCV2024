from torchvision import transforms 
import torchvision
from torchvision.transforms.autoaugment import AutoAugment as autoaug
from torchvision.transforms.autoaugment import AutoAugmentPolicy as autoaugpolicy
import torch.nn as nn
import torch
import numpy as np


class PretrainTransform:
    def __init__(self, dataset, args):
        self.args = args
        if dataset=="cifar100":
            self.data_normalize_mean = (0.5071, 0.4865, 0.4409)
            self.data_normalize_std = (0.2673, 0.2564, 0.2762)
            self.transform = transforms.Compose([
                transforms.RandomChoice([transforms.RandomCrop(32, padding=4),transforms.RandomResizedCrop(32, scale=(0.25,1), ratio=(1,1))]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([autoaug(autoaugpolicy.CIFAR10,transforms.InterpolationMode.BILINEAR)],p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(self.data_normalize_mean, self.data_normalize_std)
            ])
        elif dataset=="mini_imagenet":
            self.data_normalize_mean = (0.485, 0.456, 0.406)
            self.data_normalize_std = (0.229, 0.224, 0.225)
            self.random_crop_size = 84
            self.min_scale = 0.08
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(84, scale=(0.08,1)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([autoaug(autoaugpolicy.IMAGENET,transforms.InterpolationMode.BILINEAR)],p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
        elif dataset=='cub200':
            self.data_normalize_mean = (0.485, 0.456, 0.406)
            self.data_normalize_std = (0.229, 0.224, 0.225)
            self.random_crop_size = 224
            self.min_scale = 0.25
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.25,1), ratio=(1,1)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            raise ValueError('Dataset is not normalized!')
        

    def __call__(self, x):
        y = []
        for _ in range(self.args.num_aug+1):
            y.append(self.transform(x))

        return y