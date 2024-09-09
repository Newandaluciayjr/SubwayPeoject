import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(image_path).convert('RGB')  # 假设掩码和图像在同一路径
        mask = rgb_to_class(mask)
        mask = Image.fromarray(mask).convert('P')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            mask = torch.tensor(mask, dtype=torch.long)
            mask = mask.squeeze(0)

        return image, mask

# 定义颜色到类别的映射
COLOR_DICT = {
    (255, 190, 0): 0,  # 红色表示人
    (0, 255, 0): 1,  # 蓝色表示家具
    (0, 0, 0): 2,    # 黑色表示看线
    (255, 255, 255): 3  # 白色表示背景
}

# 将RGB掩码图像转换为类别掩码
def rgb_to_class(mask):
    mask = np.array(mask)
    class_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    for rgb, cls in COLOR_DICT.items():
        class_mask[(mask == rgb).all(axis=2)] = cls
    return class_mask

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.long()),  # 转换为长整型
])

# 假设 'dataset/png/' 文件夹中包含了所有的 PNG 图像
image_folder = 'dataset/png/'
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')]

# 创建数据集实例
dataset = CustomDataset(image_paths=image_paths, transform=transform)

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

