from __future__ import print_function

import torch
import torch.nn.functional as F
from models import LeNet, VGG
from torch.autograd.gradcheck import zero_gradients
from torchvision import datasets, transforms
from torchvision import models
import matplotlib.pyplot as plt
import time
import numpy as np

# 改batch_size看看

alpha = 0.03
epsilons = [0.6]
iter_num = 12
edit_point_num = 3
target_num = 0

image_size = 32
channel_size = 3

# image_size = 28
# channel_size = 1

pretrained_model = "./lenet_mnist_model.pth"
use_cuda = True



def main():
    # Set CUDA
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # Dataloaders
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
    #         transforms.ToTensor(),
    #     ])),
    #     batch_size=1, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=1, shuffle=False)

    # Model
    # model = LeNet().to(device)
    # model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
    # model.eval()

    model = torch.load(f'./VGG19.pth').to(device)
    model.eval()



    accuracies = []
    examples = []
    cleans = []
    total_grads = []

    for eps in epsilons:
        acc, ex, cl, grads = test(model, device, test_loader, eps)
        accuracies.append(acc)
        examples.append(ex)
        cleans.append(cl)
        total_grads.append(grads)

    for i in range(0, len(examples[0])):
        orig, adv, ex = examples[0][i]
        cl = cleans[0][i]
        grad = total_grads[0][i]
        visualize(cl, ex, grad, alpha, orig, adv)


def test(model, device, test_loader, epsilon):
    tstart = time.time()
    # Accuracy counter
    correct = 0
    adv_examples = []
    cl_examples = []
    grads = []

    target_fake1 = torch.tensor([target_num]).to(device)
    target_fake1.requires_grad = False

    # Loop over all examples in test set
    for step, (data, target) in enumerate(test_loader):
        if step > 10000: break
        # Send the data and label to the device

        data, target = data.to(device), target.to(device)
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True
        target.requires_grad = False


        # print('Gounded Truth: ', target.item())

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue
        if target_fake1.item() == target.item():
            continue

        topk_index = []
        image_tensor = data.data.clone()
        total_g = torch.zeros(data.size()).to(device)

        for i in range(0, iter_num):
            zero_gradients(data)
            output = model(data)

            loss = F.nll_loss(output, target_fake1)
            loss.backward(retain_graph=False)

            data_grad = data.grad.data

            if i == 0:
                data_grad_r = data_grad.clone().reshape(-1)
                data_grad_abs = torch.abs(data_grad_r)
                topk = torch.topk(data_grad_abs, edit_point_num)
                topk_index = topk[1]

            # Call FGSM Attack
            perturbed_data, g = fgsm_attack(data, alpha, data_grad, topk_index, i)

            total_grad = perturbed_data - image_tensor
            total_grad = torch.clamp(total_grad, -epsilon, epsilon)
            adv = image_tensor + total_grad
            total_g += total_grad
            data.data = adv


        # Check for success
        final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        elif final_pred.item() == target_fake1.item():
            # Save some adv examples for visualization later
            if len(adv_examples) < 100:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
                clean = image_tensor.squeeze().detach().cpu().numpy()
                cl_examples.append((clean))
                grads.append((total_g))

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(test_loader))
    print("Alpha: {}\tTest Accuracy = {} / {} = {}".format(alpha, correct, len(test_loader), final_acc))

    tend = time.time()
    print(tend - tstart)
    return final_acc, adv_examples, cl_examples, grads


# FGSM attack code
def fgsm_attack(image, alpha, data_grad, topk_index, t):
    momentum = 0.9
    sign_data_grad = data_grad.sign()

    perturbed_image = image.clone()
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    g = torch.zeros(perturbed_image.size()).to(device)
    v = 0
    for i in range(0, len(topk_index)) :
        l = topk_index[i] / channel_size
        c = l % channel_size
        m = l / image_size
        n = l % image_size
        v = (momentum * v) + (alpha * sign_data_grad[0][c][m][n])
        g[0][c][m][n] = g[0][c][m][n] - v

    perturbed_image.data = perturbed_image.data + g.data

    # perturbed_image = perturbed_image + alpha * sign_data_grad

    return perturbed_image, g


def fgsm_attack2(image, alpha, data_grad, topk_index):
    sign_data_grad = data_grad.sign()

    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image.clone()
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    g = torch.zeros(perturbed_image.size()).to(device)

    perturbed_image = perturbed_image - alpha * sign_data_grad

    return perturbed_image, g


def visualize(x, x_adv, x_grad, epsilon, clean_pred, adv_pred):
    x_grad = x_grad.detach().cpu().squeeze().numpy()

    figure, ax = plt.subplots(1, 3, figsize=(18, 8))
    x.reshape(image_size, image_size, channel_size)

    ax[0].imshow((np.transpose(x, (1, 2, 0)) * 255).astype(np.uint8))
    ax[0].set_title('Clean Example', fontsize=20)

    ax[1].imshow((np.transpose(x_grad, (1, 2, 0)) * 255).astype(np.uint8))
    ax[1].set_title('Perturbation', fontsize=20)
    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    ax[2].imshow((np.transpose(x_adv, (1, 2, 0)) * 255).astype(np.uint8))
    ax[2].set_title('Adversarial Example', fontsize=20)

    ax[0].axis('off')
    ax[2].axis('off')

    ax[0].text(1.1, 0.5, "+{}*".format(round(epsilon, 3)), size=15, ha="center",
               transform=ax[0].transAxes)

    ax[0].text(0.5, -0.13, "Prediction: {}\n".format(clean_pred), size=15, ha="center",
               transform=ax[0].transAxes)

    ax[1].text(1.1, 0.5, " = ", size=15, ha="center", transform=ax[1].transAxes)

    ax[2].text(0.5, -0.13, "Prediction: {}\n".format(adv_pred), size=15, ha="center",
               transform=ax[2].transAxes)

    plt.show()

    
if __name__ == '__main__':
    main()
