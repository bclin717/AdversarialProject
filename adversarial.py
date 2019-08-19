from __future__ import print_function

import torchvision
from torch.autograd.gradcheck import zero_gradients
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import numpy as np
import torch.backends.cudnn as cudnn
from models import *

# Configuration
from utils import UnNormalize

alpha_LL = 4
alpha_FGSM = 6
epsilons = [10]
iter_num_LL = 10
iter_num_FGSM = 10
edit_point_num_LL = 2
edit_point_num_FGSM = 2
target_nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
momentum = 0.9
count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

batch_size = 100
attack_method = "ITER"


# Set CUDA
use_cuda = True
print("CUDA Available: ", torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'

pretrained_model = "./trained_models/lenet_mnist_model.pth"
dataset = "CIFAR10"
shuffle = False

save_pics = True

unnorm = UnNormalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
# Transform
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Dataloader
if dataset == 'MNIST':
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=batch_size, shuffle=shuffle)
elif dataset == 'CIFAR10':
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, num_workers=8)

# Model
if dataset == 'MNIST':
    model = LeNet().to(device)
    model.load_state_dict(torch.load(pretrained_model))
elif dataset == 'CIFAR10':
    model = ResNeXt29_2x64d()
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    checkpoint = torch.load('./trained_models/ResNeXt29_2x64d_Weak.pth')
    model.load_state_dict(checkpoint['net'])

if dataset == 'MNIST':
    image_size = 28
    channel_size = 1
elif dataset == 'CIFAR10':
    image_size = 32
    channel_size = 3

if dataset == 'MNIST':
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
elif dataset == 'CIFAR10':
    labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']


def main():
    model.eval()

    accuracies = []
    examples = []
    cleans = []
    total_grads = []

    # testing
    for eps in epsilons:
        for target_num in target_nums:
            acc, ex, cl, grads = test(model, device, test_loader, eps, target_num)
            accuracies.append(acc)
            examples.append(ex)
            cleans.append(cl)
            total_grads.append(grads)


def test(model, device, test_loader, epsilon, target_num):
    tstart = time.time()
    # Accuracy counter
    correct = 0
    adv_success = 0
    incorrect = 0
    org_incorrect = 0

    adv_examples = []
    cl_examples = []
    grads = []

    target_fake1 = torch.tensor([target_num]).to(device)
    target_fake1.requires_grad = False
    count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # Loop over all examples in test set
    for step, (dataAll, targetAll) in enumerate(test_loader):
        for batch in range(0, batch_size):
            data, target = dataAll[batch], targetAll[batch]
            data, target = data.unsqueeze(0).to(device), target.unsqueeze(0).to(device)
            data.requires_grad = True
            target.requires_grad = False

            # Forward pass the data through the model
            output = model(data)
            init_pred = output.max(1, keepdim=True)[1]

            if init_pred.item() != target.item():
                org_incorrect += 1
                continue
            elif target_fake1.item() == target.item():
                continue

            topk_index = []
            topk_index2 = []
            image_tensor = data.data.clone()

            for i in range(0, iter_num_LL):
                zero_gradients(data)
                output = model(data)
                pred = output.max(1, keepdim=True)[1]
                if pred == target_fake1:
                    break
                loss = F.nll_loss(output, target_fake1)
                loss.backward(retain_graph=False)

                data_grad = data.grad.data

                # Top K
                if i == 0:
                    data_grad_r = data_grad.clone().reshape(-1)
                    data_grad_abs = torch.abs(data_grad_r)
                    topk = torch.topk(data_grad_abs, edit_point_num_LL)
                    topk_index = topk[1]

                adv = iter_attack_topK_sourceTargeting(data, data_grad, image_tensor, topk_index, epsilon)
                data.data = adv

            for i in range(0, iter_num_FGSM):
                zero_gradients(data)
                output = model(data)
                pred = output.max(1, keepdim=True)[1]
                if pred == target_fake1:
                    break
                loss = F.nll_loss(output, target)
                loss.backward(retain_graph=False)

                data_grad = data.grad.data

                # Top K
                if i == 0:
                    data_grad_r = data_grad.clone().reshape(-1)
                    data_grad_abs = torch.abs(data_grad_r)
                    topk = torch.topk(data_grad_abs, edit_point_num_FGSM)
                    topk_index2 = topk[1]

                # Attack
                adv = fgsm_attack_topK(adv, data_grad, topk_index2)
                data.data = adv

            total_g = adv - image_tensor

            # Check for success
            output = model(data)
            final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            if final_pred.item() == target.item():
                correct += 1
            elif final_pred.item() == target_fake1.item():
                adv_success += 1
                if save_pics:
                    name = "./adv/adv_" + "batch" + str(step) + "_" + str(batch) + "_" + labels[target.item()] + "To" + \
                           labels[target_fake1.item()] + ".png"
                    torchvision.utils.save_image(unnorm(adv), filename=name)
                # Save some adv examples for visualization later
                if len(adv_examples) < 100:
                    adv_ex = adv.squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
                    clean = image_tensor.squeeze().detach().cpu().numpy()
                    cl_examples.append(clean)
                    grads.append(total_g)

            if final_pred.item() != target.item():
                count[final_pred] += 1
                incorrect += 1

    # Calculate final accuracy for this epsilon
    allnum = step * (batch_size + 1)
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
    print("Cost time : {:.2f} seconds".format(tend - tstart))
    print("")

    return final_acc, adv_examples, cl_examples, grads


# FGSM attack code
def iter_attack_topK_sourceTargeting(image, data_grad, image_tensor, topk_index, epsilon):
    sign_data_grad = data_grad.sign()

    perturbed_image = image.clone()
    g = torch.zeros(perturbed_image.size()).to(device)
    v = 0
    for i in range(0, len(topk_index)):
        l = topk_index[i] / channel_size
        c = l % channel_size
        m = l / image_size
        n = l % image_size
        v = (momentum * v) + (alpha_LL * sign_data_grad[0][c][m][n])
        g[0][c][m][n] = g[0][c][m][n] - v

    perturbed_image.data = perturbed_image.data + g.data
    total_grad = perturbed_image - image_tensor
    total_grad = torch.clamp(total_grad, -epsilon, epsilon)
    adv = image_tensor + total_grad

    return adv


def fgsm_attack(image, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image.clone()
    g = torch.zeros(perturbed_image.size()).to(device)
    perturbed_image = perturbed_image + alpha_FGSM * sign_data_grad
    return perturbed_image


def fgsm_attack_topK(image, data_grad, topk_index):
    sign_data_grad = data_grad.sign()
    perturbed_image = image.clone()
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

if __name__ == '__main__':
    main()
