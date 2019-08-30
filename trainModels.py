'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--test', action='store_true')
parser.add_argument('--retrain', action='store_true')
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
lrs = [0.01, 0.001]
batch_size = 256
shuffle = True

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    if args.retrain:
        best_acc = 0
        start_epoch = 0
    else:
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


def ExtractDatasetSortedByLabels(train=True):
    if train:
        name = "TrainSet"
    else:
        name = "TestSet"
    path = "./Clean_CIFAR10/" + name + '/'
    dataset = torchvision.datasets.CIFAR10(root="./Clean_CIFAR10/", train=train, download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor()
                                           ]))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)

    for i in range(0, 10):
        if not os.path.isdir(path + labels[i]):
            os.makedirs(path + labels[i])

    print('test set:', len(trainloader))
    f = open('./Clean_CIFAR10/' + name + '/CIFAR10_Labels.txt', 'w')
    for i, (img, label) in enumerate(loader):
        img.to(device)
        img_path = "./Clean_CIFAR10/" + str(name) + "/" + labels[label] + "/" + str(i) + ".jpg"
        torchvision.utils.save_image(img, filename=img_path)
        f.write(img_path + ' ' + str(label.item()) + '\n')
    f.close()


def CustomDataset():
    global trainloader
    global testloader
    train_path = "./Clean_CIFAR10/TrainSet/"
    test_path = "./Clean_CIFAR10/TestSet/"
    trainset = torchvision.datasets.ImageFolder(train_path, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8,
                                              pin_memory=True)

    testset = torchvision.datasets.ImageFolder(test_path, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=8,
                                             pin_memory=True)


def NormalDataset():
    global trainloader
    global testloader
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8,
                                              pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8,
                                             pin_memory=True)

def main():
    CustomDataset()
    global start_epoch
    global optimizer

    if args.test:
        test(0)
    else:
        for i in range(0, len(lrs)):
            optimizer = optim.SGD(net.parameters(), lr=lrs[i], momentum=0.9, weight_decay=5e-4)
            for epoch in range(start_epoch, 75):
                train(epoch)
                test(epoch)
            start_epoch = 0




if __name__ == '__main__':
    main()
