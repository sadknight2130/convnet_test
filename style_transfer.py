from __future__ import division
from torchvision import models
from torchvision import transforms
from PIL import Image
import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from data_utils import load_image
from model_pretrain import VGGNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.cuda.FloatTensor
# 载入图片
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])  # 来自ImageNet的mean和variance
unloader = transforms.ToPILImage()  # reconvert into PIL image

style = load_image("png/starry_night.jpg", transform, max_size=400)
content = load_image("png/tubingen.jpg", transform, shape=[style.size(2), style.size(3)])

# 加载模型

target = content.clone().requires_grad_(True)   # 输出图像
optimizer = torch.optim.Adam([target], lr=0.003, betas=(0.5, 0.999))
vgg = VGGNet().to(device).eval()


# 训练
total_step = 2000
content_weight = 1.
tv_weight = 1.0
style_weight = 100.
for step in range(total_step):
    target_features = vgg(target)
    content_features = vgg(content)
    style_features = vgg(style)

    style_loss = 0
    content_loss = 0
    tv_loss = 0
    for f1, f2, f3 in zip(target_features, content_features, style_features):
        # content loss计算
        content_loss += torch.mean((f1 - f2) ** 2)

        # total variation loss 计算
        _, c, h, w = f1.size()
        diff_H = f2[:, :, 1:, :] - f2[:, :, :-1, :]
        diff_W = f2[:, :, :, 1:] - f2[:, :, :, :-1]
        tv_loss += torch.sum(torch.square(diff_H)) + torch.sum(torch.square(diff_W))

        # 计算gram matrix
        f1 = f1.view(c, h * w)
        f3 = f3.view(c, h * w)
        f1 = torch.mm(f1, f1.t())  # C*C
        f3 = torch.mm(f3, f3.t())  # C*C
        # style loss计算
        style_loss += torch.mean((f1 - f3) ** 2) / (c * h * w)

    loss = content_weight * content_loss + style_weight * style_loss + tv_weight * tv_loss

    # 更新target
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print("Step [{}/{}], Content Loss: {:.4f}, Style Loss: {:.4f}, Tv Loss: {:.4f}"
              .format(step, total_step, content_loss.item(), style_loss.item(), tv_loss.item()))