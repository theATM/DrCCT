import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import joblib
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from Res2Net.train import validate

from utils import *
import res2net


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--dataset', default='cifar100',
                        choices=['cifar100'],
                        help='dataset name')
    parser.add_argument('--arch', default='res2next29_6cx24wx6scale_se',
                        choices=res2net.__all__,
                        help='model architecture')
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--nesterov', default=False, type=str2bool)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args.name is None:
        args.name = '%s' %args.arch


    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True

    # data loading code
    if args.dataset == 'cifar100':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        test_set = datasets.CIFAR100(
            root='~/data',
            train=False,
            download=True,
            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=32,
            shuffle=False,
            num_workers=8)

        num_classes = 100

    # create model
    model = res2net.__dict__[args.arch]()
    model = model.cuda()
    if args.checkpoint != '':
        model.load_state_dict(torch.load(args.checkpoint))
    else:
        print("No model weights were loaded please add --checkpoint")

    # evaluate on validation set
    val_log = validate(args, test_loader, model, criterion)
    print('Validation on testset:'
          'val_loss %.4f - val_acc %.4f - val_acc5 %.4f'
        %(val_log['loss'], val_log['acc1'], val_log['acc5']))



if __name__ == '__main__':
    main()