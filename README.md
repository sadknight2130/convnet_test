# convnet_test
first try
1.测试CNN卷积神经网络在MNIST的应用，主要是熟悉pytorch用法，掌握用pytorch训练网络
2.用预训练的模型来训练，掌握两种预训练的方法（fine tunning, feature extract）

文件结构
            
            --MNIST

            --hymenoptera_data
            
            --xx.py
            
            --xxx.py

# StyleTransfer


风格迁移是一个很有意思的任务，通过风格迁移可以使一张图片保持本身内容大致不变的情况下呈现出另外一张图片的风格。

* 固定风格固定内容的普通风格迁移（[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)）

环境是 pytorch 0.4.0

# 固定风格固定内容的普通风格迁移

最早的风格迁移就是在固定风格、固定内容的情况下做的风格迁移，这是最慢的方法，也是最经典的方法。

最原始的风格迁移的思路很简单，把图片当做可以训练的变量，通过优化图片来降低与内容图片的内容差异以及降低与风格图片的风格差异，迭代训练多次以后，生成的图片就会与内容图片的内容一致，同时也会与风格图片的风格一致。

## VGG19

VGG19 是一个很经典的模型，它通过堆叠 3x3 的卷积层和池化层，在 ImageNet 上获得了不错的成绩。我们使用在 ImageNet 上经过预训练的 VGG16 模型可以对图像提取出有用的特征，这些特征可以帮助我们去衡量两个图像的内容差异和风格差异。

在进行风格迁移任务时，我们只需要提取其中几个比较重要的层，所以我们对 pytorch 自带的预训练 VGG16 模型稍作了一些修改：

```py
class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.select = ['0', '5', '10', '19', '28']
        self.vgg = models.vgg19(pretrained=True).features

    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features
        
        
target = content.clone().requires_grad_(True)   # 输出图像
optimizer = torch.optim.Adam([target], lr=0.003, betas=(0.5, 0.999))
vgg = VGGNet().to(device).eval()
```

经过修改的 VGG19 可以输出 0, 5, 10, 19, 28 这几个特定层的特征图。下面这两句代码就是它的用法：

```py
target_features = vgg(target)
content_features = vgg(content)
style_features = vgg(style)
```

举个例子，当我们使用 vgg19 对 `target` 计算特征时，它会返回五个矩阵给 features，假设 `target` 的尺寸是 `[1, 3, 400, 316]`（四个维度分别代表 batch, channels, height, width），那么它返回的四个矩阵的尺寸就是这样的：

* 0 `[1, 64, 400, 316]`
* 5 `[1, 128, 200, 158]`
* 10 `[1, 256, 100, 79]`
* 19 `[1, 512, 50, 39]`
* 28 `[1, 512, 25, 19]`

## 内容

我们进行风格迁移的时候，必须保证生成的图像与内容图像的内容一致性，通过 VGG16 输出的特征图来衡量图片的内容差异。

这里使用的损失函数是：

![equation](https://latex.codecogs.com/svg.latex?$$\Large\ell^{\phi,j}_{feat}(\hat{y},y)=\frac{1}{C_jH_jW_j}||\phi_j(\hat{y})-\phi_j(y)||^2_2$$)


其中：

* ![equation](https://latex.codecogs.com/svg.latex?\hat{y})是输入图像（也就是生成的图像）
* ![equation](https://latex.codecogs.com/svg.latex?y)是内容图像
* ![equation](https://latex.codecogs.com/svg.latex?\phi) 代表 VGG16
* ![equation](https://latex.codecogs.com/svg.latex?\j) 在这里是 relu3_3
* ![equation](https://latex.codecogs.com/svg.latex?\phi_j(x))指的是 x 图像输入到 VGG 以后的第 j 层的特征图
* ![equation](https://latex.codecogs.com/svg.latex?C_j\times&space;H_j\times&space;W_j)是第 j 层输出的特征图的尺寸

根据生成图像和内容图像在 ![equation](https://latex.codecogs.com/svg.latex?$\text{relu3\_3}$) 输出的特征图的均方误差（MeanSquaredError）来优化生成的图像与内容图像之间的内容一致性。


那么写成代码就是这样的：

```py
content_loss = torch.mean((target_features[2]-content_features[2])**2)
```

因为我们这里使用的是经过在 ImageNet 预训练过的 VGG19 提取的特征图，所以它能提取出图像的高级特征，通过优化生成图像和内容图像特征图的 mse，可以迫使生成图像的内容与内容图像在 VGG16 的 relu3\_3 上输出相似的结果，因此生成图像和内容图像在内容上是一致的。

## 风格

### Gram 矩阵

采用Gram 矩阵衡量风格：

![equation](https://latex.codecogs.com/svg.latex?$$\Large{G^\phi_j(x)_{c,c'}=\frac{1}{C_jH_jW_j}&space;\sum_{h=1}^{H_j}&space;\sum_{w=1}^{W_j}&space;\phi_j(x)_{h,w,c}\phi_j(x)_{h,w,c'}}$$)

其中：

* ![equation](https://latex.codecogs.com/svg.latex?\hat{y})是输入图像（也就是生成的图像）
* ![equation](https://latex.codecogs.com/svg.latex?y)是风格图像
* ![equation](https://latex.codecogs.com/svg.latex?C_j\times&space;H_j\times&space;W_j)是第 j 层输出的特征图的尺寸。
* ![equation](https://latex.codecogs.com/svg.latex?$G^\phi_j(x)$)指的是 x 图像的第 j 层特征图对应的 Gram 矩阵，比如 64 个卷积核对应的卷积层输出的特征图的 Gram 矩阵的尺寸是 ![equation](https://latex.codecogs.com/svg.latex?$(64,64)$)。
* ![equation](https://latex.codecogs.com/svg.latex?$G^\phi_j(x)_{c,c'}$) 指的是 Gram 矩阵第 ![equation](https://latex.codecogs.com/svg.latex?$(c,c')$) 坐标对应的值。
* ![equation](https://latex.codecogs.com/svg.latex?$\phi_j(x)$)指的是 x 图像输入到 VGG 以后的第 j 层的特征图，![equation](https://latex.codecogs.com/svg.latex?$\phi_j(x)_{h,w,c}$) 指的是特征图 ![equation](https://latex.codecogs.com/svg.latex?$(h,w,c)$)坐标对应的值。

Gram 矩阵的计算方法其实很简单，Gram 矩阵的 ![equation](https://latex.codecogs.com/svg.latex?$(c,c')$) 坐标对应的值，就是特征图的第 ![equation](https://latex.codecogs.com/svg.latex?$c$) 张和第 ![equation](https://latex.codecogs.com/svg.latex?$c'$) 张图对应元素相乘，然后全部加起来并且除以 ![equation](https://latex.codecogs.com/svg.latex?C_j\times&space;H_j\times&space;W_j) 的结果。根据公式我们可以很容易推断出 Gram 矩阵是对称矩阵。

具体到代码，我们可以写出下面的函数：

```py
def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram
```


### 风格损失

根据生成图像和风格图像在 relu1_2、relu2_2、relu3_3、relu4_3 输出的特征图的 Gram 矩阵之间的均方误差（MeanSquaredError）来优化生成的图像与风格图像之间的风格差异：

![equation](https://latex.codecogs.com/svg.latex?$$\Large\ell^{\phi,j}_{style}(\hat{y},y)=||G^\phi_j(\hat{y})-G^\phi_j(y)||^2_F$$)

其中：

* ![equation](https://latex.codecogs.com/svg.latex?\hat{y})是输入图像（也就是生成的图像）
* ![equation](https://latex.codecogs.com/svg.latex?$y$)是风格图像
* ![equation](https://latex.codecogs.com/svg.latex?$G^\phi_j(x)$)指的是 x 图像的第 j 层特征图对应的 Gram 矩阵

那么写成代码就是下面这样：

```py
for  f1, f3 in zip(target_features, style_features):
        
        _, c, h, w = f1.size()
        f1 = f1.view(c, h*w)
        f3 = f3.view(c, h*w)
        
        # 计算gram matrix
        f1 = torch.mm(f1, f1.t()) # C*C
        f3 = torch.mm(f3, f3.t()) # C*C
        style_loss += torch.mean((f1-f3)**2)/(c*h*w)
        
    loss = content_loss + style_weight * style_loss
```

## 训练

那么风格迁移的目标就很简单了，直接将两个 loss 按权值加起来，然后对图片优化 loss，即可优化出既有内容图像的内容，也有风格图像的风格的图片。代码如下：

```py
total_step = 2000
style_weight = 200.
for step in range(total_step):
    target_features = vgg(target)
    content_features = vgg(content)
    style_features = vgg(style)
    
    style_loss = 0

    content_loss = torch.mean((target_features[2]-content_features[2])**2)
    for  f1, f3 in zip(target_features, style_features):
        
        _, c, h, w = f1.size()
        f1 = f1.view(c, h*w)
        f3 = f3.view(c, h*w)
        
        # 计算gram matrix
        f1 = torch.mm(f1, f1.t()) # C*C
        f3 = torch.mm(f3, f3.t()) # C*C
        style_loss += torch.mean((f1-f3)**2)/(c*h*w)
        
    loss = content_loss + style_weight * style_loss
    
    # 更新target
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
        print("Step [{}/{}], Content Loss: {:.4f}, Style Loss: {:.4f}"
             .format(step, total_step, content_loss.item(), style_loss.item()))  
```

