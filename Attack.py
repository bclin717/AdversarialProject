import os
import sys
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.utils
from torch.backends import cudnn
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from torchattacks import *

from models import *
import matplotlib.pyplot as plt

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

batch_size = 1

cifar10_train = torchvision.datasets.ImageFolder(root='./Clean_CIFAR10_For_Adv/TrainSet/', transform=transform)
cifar10_test = dsets.CIFAR10(root='./data', train=False,
                             download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=batch_size,
                                           shuffle=False, num_workers=1)

test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=batch_size,
                                          shuffle=False, num_workers=1, pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define models
use_cuda = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_url = './trained_models/VGG19_Retrained.pth'
model = VGG('VGG19')
model = model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
checkpoint = torch.load(model_url)
model.load_state_dict(checkpoint['net'])
model.eval()

count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

fgsm_attack = FGSM(model, eps=0.025)
ifgsm_attack = IFGSM(model, eps=4 / 255, alpha=1 / 255, iters=0)
iterll_attack = IterLL(model, eps=4 / 255, alpha=1 / 255, iters=0)
cw_attack = CW(model, targeted=False, c=1, kappa=0, iters=1000, lr=0.01)
pgd_attack = PGD(model, eps=0.3, alpha=2 / 255, iters=5)

dif_attack = DIF(model, eps=1, alpha=26 / 255, iters=5)

pgd = PGD(model, eps=0.3, alpha=2 / 255, iters=5)

attacks = [pgd]

print("Attack Image & Predicted Label")

model_url = './trained_models/VGG19.pth'
vgg = VGG('VGG19')
vgg = vgg.to(device)
if device == 'cuda':
    vgg = torch.nn.DataParallel(vgg)
    cudnn.benchmark = True
checkpoint = torch.load(model_url)
vgg.load_state_dict(checkpoint['net'])
vgg.eval()

model_url = './trained_models/VGG19_Retrained.pth'
vgg_r = VGG('VGG19')
vgg_r = vgg_r.to(device)
if device == 'cuda':
    vgg_r = torch.nn.DataParallel(vgg_r)
    cudnn.benchmark = True
checkpoint = torch.load(model_url)
vgg_r.load_state_dict(checkpoint['net'])
vgg_r.eval()

model_url = './trained_models/ResNet18_Strong.pth'
resnet18 = ResNet18()
resnet18 = resnet18.to(device)
if device == 'cuda':
    resnet18 = torch.nn.DataParallel(resnet18)
    cudnn.benchmark = True
checkpoint = torch.load(model_url)
resnet18.load_state_dict(checkpoint['net'])
resnet18.eval()

model_url = './trained_models/ResNet18_Retrained.pth'
resnet18_r = ResNet18()
resnet18_r = resnet18_r.to(device)
if device == 'cuda':
    resnet18_r = torch.nn.DataParallel(resnet18_r)
    cudnn.benchmark = True
checkpoint = torch.load(model_url)
resnet18_r.load_state_dict(checkpoint['net'])
resnet18_r.eval()

model_url = './trained_models/ResNeXt29_2x64d_Strong.pth'
rx29 = ResNeXt29_2x64d()
rx29 = rx29.to(device)
if device == 'cuda':
    rx29 = torch.nn.DataParallel(rx29)
    cudnn.benchmark = True
checkpoint = torch.load(model_url)
rx29.load_state_dict(checkpoint['net'])
rx29.eval()

model_url = './trained_models/ResNeXt29_2x64d_Retrained.pth'
rx29_r = ResNeXt29_2x64d()
rx29_r = rx29_r.to(device)
if device == 'cuda':
    rx29_r = torch.nn.DataParallel(rx29_r)
    cudnn.benchmark = True
checkpoint = torch.load(model_url)
rx29_r.load_state_dict(checkpoint['net'])
rx29_r.eval()

model_url = './trained_models/EfficientNetB0_Strong.pth'
eb0 = EfficientNetB0()
eb0 = eb0.to(device)
if device == 'cuda':
    eb0 = torch.nn.DataParallel(eb0)
    cudnn.benchmark = True
checkpoint = torch.load(model_url)
eb0.load_state_dict(checkpoint['net'])
eb0.eval()

model_url = './trained_models/EfficientNetB0_Retrained.pth'
eb0_r = EfficientNetB0()
eb0_r = eb0_r.to(device)
if device == 'cuda':
    eb0_r = torch.nn.DataParallel(eb0_r)
    cudnn.benchmark = True
checkpoint = torch.load(model_url)
eb0_r.load_state_dict(checkpoint['net'])
eb0_r.eval()

models = [vgg, vgg_r, resnet18, resnet18_r, rx29, rx29_r, eb0, eb0_r]

for model in models:
    for attack in attacks:
        print("-" * 70)
        print(attack)

        incorrect = 0
        org_incorrect = 0
        total = 0
        st = 0
        tstart = time.time()
        # count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # target_fake = torch.tensor([i]).cuda()
        for step, (images, labels) in enumerate(test_loader):
            # if labels.item() == target_fake.item():
            #     correct += 1
            #     continue

            outputs = model(images)
            _, pre = torch.max(outputs.data, 1)
            total += 1
            if pre.item() != labels.item():
                org_incorrect += 1
                continue

            images = attack(images, labels)
            labels = labels.to(device)
            outputs = model(images)
            _, pre = torch.max(outputs.data, 1)

            if pre.item() != labels.item():
                incorrect += 1

            # if pre.item() == target_fake.item():
            #     st += 1
            # if pre.item() != labels.item():
            #     count[pre.item()] += 1

            # print("The target is ", classes[i])
        # print(count)
        print('Misclassification of test text: %f %%' % (100 * float(incorrect) / (total - org_incorrect)))
        # print('Accuracy of ST: %f %%' % (100 * float(st) / total))
        tend = time.time()
        print("Cost time : {:.2f} seconds".format(tend - tstart))
        print("")
