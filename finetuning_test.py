import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
import time
import os
import copy

from model_pretrain import *
from train_eval import *
from data_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = ".\hymenoptera_data"
model_name = "resnet"
num_classes = 2
batch_size = 32
num_epochs = 15
feature_extract = True
input_size = 224
mode_dict = ["train", "val"]

# Load数据
dataloaders_dict = load_data(data_dir, input_size, batch_size, mode_dict)

# 加载模型
model_ft, input_size = initialize_model(model_name,
                                        num_classes, feature_extract, use_pretrained=True)
model_ft = model_ft.to(device)
# 定义优化器和loss函数
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                   model_ft.parameters()), lr=0.001, momentum=0.9)  # 只更新requires_grad = True的参数
loss_fn = nn.CrossEntropyLoss()
# 训练模型 返回最终模型和val上的准确率数据
_, ohist = train_model(model_ft, dataloaders_dict, loss_fn, optimizer, num_epochs=num_epochs)

# 加载 和 训练 fine tuning模型
model_scratch, _ = initialize_model(model_name,
                                    num_classes, feature_extract=False, use_pretrained=False)
model_scratch = model_scratch.to(device)
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                   model_scratch.parameters()), lr=0.001, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()
_, scratch_hist = train_model(model_scratch, dataloaders_dict, loss_fn, optimizer, num_epochs=num_epochs)

# 画出feature extract 和 fine tuning 模型在val上的准确率曲线
plt.title("Validation Accuracy vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Validation Accuracy")
plt.plot(range(1, num_epochs + 1), ohist, label="Pretrained")
plt.plot(range(1, num_epochs + 1), scratch_hist, label="Scratch")
plt.ylim((0, 1.))
plt.xticks(np.arange(1, num_epochs + 1, 1.0))
plt.legend()
plt.show()
