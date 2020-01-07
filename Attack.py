import os
import sys
import time

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.utils
from torch.backends import cudnn
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from torchattacks import *
from utils import UnNormalize

from models import *
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--t', default=0, type=int, help='target label')
args = parser.parse_args()


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

unnorm = UnNormalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))

batch_size = 1

cifar10_train = torchvision.datasets.ImageFolder(root='./Clean_CIFAR10_For_Adv/TrainSet', transform=transform)
cifar10_test = dsets.CIFAR10(root='./data', train=False,
                             download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=batch_size,
                                          shuffle=False, pin_memory=True)

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


path = "./Adv_CIFAR10/"
if not os.path.isdir(path):
    os.makedirs(path)



# Define models
use_cuda = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_url = './trained_models/VGG19.pth'
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
iterll_attack = IterLL(model, eps=8 / 255, alpha=0.001, iters=50)
cw_attack = CW(model, targeted=True, c=1e-2, kappa=0, iters=1500, lr=0.05)
pgd_attack = PGD(model, eps=0.3, alpha=2 / 255, iters=5)
dif_attack = DIF(model, eps=2, alpha=52 / 255, iters=10)
pgd = PGD(model, eps=0.3, alpha=2 / 255, iters=5)

i1 = IterLL(model, eps=1, alpha=0.5, iters=10)
attacks = [i1]

# f1 = FGSM(model, eps=0.01)
# f2 = FGSM(model, eps=0.005)
# f3 = FGSM(model, eps=0.001)
# attacks = [f1, f2, f3]

print("Attack Image & Predicted Label")

model_url = './trained_models/VGG19.pth'
vgg = VGG('VGG19')
vgg = vgg.to(device)
if device == 'cuda':
    vgg = torch.nn.DataParallel(vgg)
    # cudnn.benchmark = True
checkpoint = torch.load(model_url)
vgg.load_state_dict(checkpoint['net'])
vgg.eval()

models = [vgg]

t = args.t
np.set_printoptions(suppress=True)
for model in models:
    for attack in attacks:
        print("-" * 70)
        print(attack)
        print("Target: ", classes[t])
        incorrect = 0
        org_incorrect = 0
        total = 0
        st = 0

        # target
        # t = 3


        tstart = time.time()
        count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ambiguous = 0
        target_fake = torch.tensor([t]).cuda()
        for step, (images, labels) in enumerate(test_loader):
            outputs = model(images)
            _, pre = torch.max(outputs.data, 1)
            total += 1

            if target_fake.item() == labels.item():
                # org_incorrect += 1
                continue

            # target or mis
            images = attack(images, target_fake)
            # images = attack(images, labels)

            labels = labels.to(device)
            outputs = model(images)
            _, pre = torch.max(outputs.data, 1)

            if pre.item() != labels.item():
                incorrect += 1
            count[pre.item()] += 1


            if pre.item() == target_fake.item():
                st += 1
                # name = "./Adv_CIFAR10/" + str(step) + "_" + classes[labels.item()] + "To" + classes[pre.item()] + ".png"
                # torchvision.utils.save_image(unnorm(images), filename=name)

            pred_x = F.softmax(outputs, dim=1)
            pred_x = pred_x.cpu().data.numpy()
            flag = True
            for i in range(0, 10):
                if pred_x[0][i] > 0.6:
                    flag = False
                    break
            if flag:
                ambiguous += 1
                print(step, ":  ", ambiguous)

        print("ambiguous:", ambiguous)
        print("The target is ", classes[t])
        print(count)
        # print('Misclassification of test text: %f %%' % (100 * float(incorrect) / (total - org_incorrect)))
        # print('Accuracy of ST: %f %%' % (100 * float(st) / total))
        tend = time.time()
        # print("Average Cost time : {:.2f} seconds".format((tend - tstart)/(step+1)))
        print("Total Cost time : {:.2f} seconds".format(tend - tstart))
        print("")
