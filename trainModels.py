import argparse
import os

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models import *
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--test', action='store_true')
parser.add_argument('--retrain', action='store_true')
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# start from epoch 0 or last checkpoint epoch
lrs = [0.01, 0.001]
batch_size = 512
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


def train(net, start_epoch, lr, best_acc):
    for epoch in range(start_epoch, 50):
        train_epoch(net, epoch, lr)
        best_acc = test_epoch(net, epoch, best_acc)
    test_epoch(net, 0, best_acc)


def train_epoch(net, epoch, lr):
    CustomDataset()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
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


def test_epoch(net, epoch, best_acc):
    NormalDataset()
    criterion = nn.CrossEntropyLoss()
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

    return acc


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
    start_epoch = 0
    best_acc = 0

    # Model
    print('==> Building model..')
    net = VGG('VGG19')
    # net = ResNet18()
    # net = ResNeXt29_2x64d()
    # net = EfficientNetB0()

    net = net.to(device)

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

    # Testing or Training
    if args.test:
        test_epoch(net, 0, best_acc)
    else:
        train(net, start_epoch, 0.001, best_acc)

if __name__ == '__main__':
    main()
