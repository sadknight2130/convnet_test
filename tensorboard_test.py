import torch
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision import transforms

writer = SummaryWriter("logs")
image_path = "data/image_transform/ytp.jpg"
img = Image.open(image_path)
print(img)

# ToTensor
img_tensor = transforms.ToTensor()(img)
writer.add_image("ToTensor", img_tensor)

# Normalize
img_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])(img_tensor)
writer.add_image("Norm", img_norm)

# resize
img_resize = transforms.Resize((224, 224))(img)
img_resize = transforms.ToTensor()(img_resize)
writer.add_image("resize", img_resize)

# compose
trans_compose = transforms.Compose([
    transforms.RandomCrop((500, 800)),
    transforms.ToTensor()
])

for i in range(10):
    img_crop = trans_compose(img)
    writer.add_image("randomcrop", img_crop, i)

writer.close()

