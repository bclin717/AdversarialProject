import torch
import torch.nn as nn

from ..attack import Attack


class DIF(Attack):

    def __init__(self, model, eps=4 / 255, alpha=1 / 255, iters=0, target=0):
        super(DIF, self).__init__("DIF", model)
        self.eps = eps
        self.target = target
        self.alpha = alpha
        if iters == 0:
            self.iters = int(min(eps * 255 + 4, 1.25 * eps * 255))
        else:
            self.iters = iters

    def forward(self, images, labels, target):
        images = images.to(self.device)
        labels = labels.to(self.device)
        target = target.to(self.device)

        loss = nn.CrossEntropyLoss()

        for i in range(self.iters):
            images.requires_grad = True
            outputs = self.model(images)
            cost = loss(outputs, target).to(self.device)

            grad = torch.autograd.grad(cost, images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = images - self.alpha * grad.sign()

            a = torch.clamp(images - self.eps, min=0)
            b = (adv_images >= a).float() * adv_images + (a > adv_images).float() * a
            c = (b > images + self.eps).float() * (images + self.eps) + (images + self.eps >= b).float() * b
            images = torch.clamp(c, max=1).detach()

        for i in range(self.iters):
            images.requires_grad = True
            outputs = self.model(images)
            cost = loss(outputs, labels).to(self.device)

            grad = torch.autograd.grad(cost, images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = images + 1 / 255 * grad.sign()

            a = torch.clamp(images - self.eps, min=0)
            b = (adv_images >= a).float() * adv_images + (a > adv_images).float() * a
            c = (b > images + self.eps).float() * (images + self.eps) + (images + self.eps >= b).float() * b
            images = torch.clamp(c, max=1).detach()

        adv_images = images

        return adv_images
