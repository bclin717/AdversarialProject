from __future__ import print_function

import os
import time

import torch.backends.cudnn as cudnn
import torchvision
from torch.autograd.gradcheck import zero_gradients
from torchvision import transforms

from models import *
from utils import UnNormalize

alpha_LL = 4
alpha_FGSM = 6
epsilons = 10
iter_num_LL = 7
iter_num_FGSM = 7
edit_point_num_LL = 3
edit_point_num_FGSM = 2
target_nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
momentum = 0.9
count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

batch_size = 1
attack_method = "ITER"
dataset = "CIFAR10"
shuffle = False
save_pics = True

# Set CUDA
use_cuda = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_url = './trained_models/VGG19_Strong.pth'

unnorm = UnNormalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
# Transform
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Dataloader
if dataset == 'CIFAR10':
    # Test unseen
    # test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_train)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=shuffle, num_workers=8)

    # 首次攻擊
    train_path = "./Clean_CIFAR10_For_Adv/TrainSet/"
    trainset = torchvision.datasets.ImageFolder(train_path, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=8,
                                               pin_memory=True)

# Model
if dataset == 'CIFAR10':
    # 每次都要改
    model = VGG('VGG19')
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    # 每次都要改
    checkpoint = torch.load(model_url)
    model.load_state_dict(checkpoint['net'])

if dataset == 'CIFAR10':
    image_size = 32
    channel_size = 3

labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

path = "./Adv_CIFAR10/"
for num in range(0, 10):
    if not os.path.isdir(path + labels[num]):
        os.makedirs(path + labels[num])


def main():
    global iter_num_FGSM
    global iter_num_LL
    model.eval()
    # testing
    for target_num in target_nums:
        test(model, device, train_loader, epsilons, target_num)


def test(model, device, test_loader, epsilon, target_num):
    tstart = time.time()
    # Accuracy counter
    correct = 0
    adv_success = 0
    incorrect = 0
    org_incorrect = 0

    target_fake = torch.tensor([target_num]).to(device)
    target_fake.requires_grad = False
    count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # Loop over all examples in test set
    for step, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        target.requires_grad = False

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]

        if init_pred.item() != target.item():
            org_incorrect += 1
            continue
        elif target_fake.item() == target.item():
            if init_pred.item() == target.item():
                correct += 1
            continue

        image_tensor = data.data.clone()

        # data = iter_ST_Attack(data, target_fake, image_tensor, epsilon)
        # data = BIM_Attack(data, target_fake, target)

        topk_index = []
        topk_index2 = []
        image_tensor = data.data.clone()

        for i in range(0, iter_num_LL):
            zero_gradients(data)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            if pred == target_fake:
                break
            loss = F.nll_loss(output, target_fake)
            loss.backward()

            data_grad = data.grad.data

            # Top K
            if i == 0:
                data_grad_r = data_grad.clone().reshape(-1)
                data_grad_abs = torch.abs(data_grad_r)
                topk = torch.topk(data_grad_abs, edit_point_num_LL)
                topk_index = topk[1]
            data.data = sourceTargetingAttack_topK(data, data_grad, image_tensor, topk_index, epsilon)

        for i in range(0, iter_num_FGSM):
            zero_gradients(data)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            if pred == target_fake:
                break
            loss = F.nll_loss(output, target)
            loss.backward()

            data_grad = data.grad.data

            # Top K
            if i == 0:
                data_grad_r = data_grad.clone().reshape(-1)
                data_grad_abs = torch.abs(data_grad_r)
                topk = torch.topk(data_grad_abs, edit_point_num_FGSM)
                topk_index2 = topk[1]
            data.data = fgsmAttack_topK(data, data_grad, topk_index2)


        # Check for success
        output = model(data)
        final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
        elif final_pred.item() == target_fake.item():
            adv_success += 1
            # print(step, "+1")
            if save_pics:
                name = "./Adv_CIFAR10/" + labels[target.item()] + "/" + "batch" + str(step) + "_Iternum_" + str(
                    iter_num_LL) + "_" + labels[
                    target.item()] + "To" + \
                       labels[target_fake.item()] + ".png"
                torchvision.utils.save_image(unnorm(data), filename=name)
        if final_pred.item() != target.item():
            count[final_pred] += 1
            incorrect += 1

    # Calculate final accuracy for this epsilon
    allnum = (step + 1)
    final_acc = correct / float(allnum)
    final_incorrect = incorrect / float(allnum - org_incorrect)
    final_adv_suc = adv_success / float(allnum - org_incorrect)

    tend = time.time()

    print(count)
    print("Number of samples: {}".format(allnum))
    print("Target: {}".format(labels[target_num]))
    print("Test Accuracy = {} / {} = {:.2%}".format(correct, allnum, final_acc))
    print("Incorrect = {} / {} = {:.2%}".format(incorrect, allnum - org_incorrect, final_incorrect))
    print("Successful Adv source-targeting = {} / {} = {:.2%}".format(adv_success, allnum - org_incorrect,
                                                                      final_adv_suc))
    print("Cost time : {:.2f} seconds".format((tend - tstart) / allnum))
    print("")


def sourceTargetingAttack_topK(data, data_grad, image_tensor, topk_index, epsilon):
    sign_data_grad = data_grad.sign()

    perturbed_image = data.clone()
    g = torch.zeros(perturbed_image.size()).to(device)
    v = 0
    for i in range(0, len(topk_index)):
        l = topk_index[i] / channel_size
        c = l % channel_size
        m = l / image_size
        n = l % image_size
        v = (momentum * v) + (alpha_LL * sign_data_grad[0][c][m][n])
        g[0][c][m][n] = g[0][c][m][n] - v

    perturbed_image.data = perturbed_image.data + g
    total_grad = perturbed_image - image_tensor
    total_grad = torch.clamp(total_grad, -epsilon, epsilon)
    data = image_tensor + total_grad

    return data


def fgsmAttack_topK(data, data_grad, topk_index):
    sign_data_grad = data_grad.sign()

    perturbed_image = data.clone()
    g = torch.zeros(perturbed_image.size()).to(device)
    v = 0
    for i in range(0, len(topk_index)):
        l = topk_index[i] / channel_size
        c = l % channel_size
        m = l / image_size
        n = l % image_size
        v = (momentum * v) + (alpha_FGSM * sign_data_grad[0][c][m][n])
        g[0][c][m][n] = g[0][c][m][n] + v

    perturbed_image.data = perturbed_image.data + g.data
    return perturbed_image


def fgsm_attack(image, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image.clone()
    g = torch.zeros(perturbed_image.size()).to(device)
    perturbed_image = perturbed_image + alpha_FGSM * sign_data_grad
    return perturbed_image


if __name__ == '__main__':
    main()
