import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
import os

from Pix2PixModel import UNetGenerator

# 定义并加载生成器
generator = UNetGenerator(in_channels=3, out_channels=3)
# torch.cuda.empty_cache()
generator.load_state_dict(torch.load('pix2pix_generator.pth'))
generator.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)

# 定义图像转换和预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# 指定输入和输出文件夹
input_folder = 'test/input'
output_folder = 'test/output'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有图片
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):  # 确保处理的是PNG图片
        input_image_path = os.path.join(input_folder, filename)
        input_image = Image.open(input_image_path).convert('RGB')

        # 应用转换和预处理
        input_image = transform(input_image).unsqueeze(0).to(device)

        # 生成图像
        with torch.no_grad():
            generated_image = generator(input_image)

        # 将生成的图像转换回PIL图像并保存
        generated_image = generated_image.squeeze(0).cpu()
        generated_image = transforms.ToPILImage()(generated_image)
        output_image_path = os.path.join(output_folder, filename)
        generated_image.save(output_image_path)

        # 可选：显示生成的图像
        plt.imshow(generated_image)
        plt.axis('off')
        plt.show()