import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import os
from torchvision import transforms

from Dataprocessing import dataloader

# 定义生成器和判别器
class UNetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetGenerator, self).__init__()
        # 定义 U-Net 的编码器和解码器
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.encoder5 = self.conv_block(512, 1024)  # 新增一层

        self.bottleneck = self.conv_block(1024, 2048)  # 新增一层

        self.decoder5 = self.upconv_block(2048, 1024)  # 新增一层
        self.decoder4 = self.upconv_block(2048, 512)  # 修改为 2048 -> 1024
        self.decoder3 = self.upconv_block(1024, 256)  # 修改为 1024 -> 512
        self.decoder2 = self.upconv_block(512, 128)  # 修改为 512 -> 256
        self.decoder1 = self.upconv_block(256, 64)  # 修改为 256 -> 128

        self.final_layer = nn.Conv2d(128, out_channels, kernel_size=1)  # 修改为 64 -> 128

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block

    def upconv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x):
        # 编码器部分
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)  # 新增一层

        b = self.bottleneck(e5)

        # 解码器部分，增加跳跃连接
        d5 = self.decoder5(b)
        d5 = torch.cat((d5, e5), dim=1)

        d4 = self.decoder4(d5)
        d4 = torch.cat((d4, e4), dim=1)

        d3 = self.decoder3(d4)
        d3 = torch.cat((d3, e3), dim=1)

        d2 = self.decoder2(d3)
        d2 = torch.cat((d2, e2), dim=1)

        d1 = self.decoder1(d2)
        d1 = torch.cat((d1, e1), dim=1)

        output = self.final_layer(d1)

        # 使用 interpolate 确保输出图像尺寸与输入图像相同
        output = torch.sigmoid(output)
        output = torch.nn.functional.interpolate(output, size=(x.size(2), x.size(3)), mode='bilinear',
                                                 align_corners=True)

        return output

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            self.conv_block(in_channels, 64, bn=False),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
            self.conv_block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def conv_block(self, in_channels, out_channels, bn=True):
        block = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
        if bn:
            block.append(nn.BatchNorm2d(out_channels))
        block.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*block)

    def forward(self, x):
        output = self.net(x)
        return torch.sigmoid(output)

# 设置模型参数
input_channels = 3
output_channels = 3
lr = 0.00005  # 学习率
num_epochs = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化生成器和判别器
generator = UNetGenerator(input_channels, output_channels).to(device)
discriminator = Discriminator(input_channels + output_channels).to(device)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# 记录损失值
g_losses = []
d_losses = []

def generate_images(model, test_anno, test_real):
    prediction = model(test_anno).permute(0, 2, 3, 1).detach().cpu().numpy()
    test_anno = test_anno.permute(0, 2, 3, 1).cpu().numpy()
    test_real = test_real.permute(0, 2, 3, 1).cpu().numpy()
    plt.figure(figsize=(10, 10))
    display_list = [test_anno[0], test_real[0], prediction[0]]
    title = ['Input', 'Ground Truth', 'Output']
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')  # 坐标系关掉

    # 保存生成图像
    plt.savefig(f'epoch/generated_images_epoch_{epoch}.png')
    plt.close()  # 关闭当前图像，防止在生成很多图像时内存占用过高

# 训练 Pix2Pix 模型
for epoch in range(num_epochs):
    g_loss_epoch = 0.0
    d_loss_epoch = 0.0
    for i, (input_image, target_image) in enumerate(tqdm(dataloader)):
        input_image = input_image.to(device)
        target_image = target_image.to(device)

        # 生成对抗样本
        fake_target_image = generator(input_image)

        # 判别器损失
        real_output = discriminator(torch.cat((input_image, target_image), dim=1))
        fake_output = discriminator(torch.cat((input_image, fake_target_image.detach()), dim=1))

        real_labels = torch.ones_like(real_output).to(device)
        fake_labels = torch.zeros_like(fake_output).to(device)

        d_loss_real = criterion(real_output, real_labels)
        d_loss_fake = criterion(fake_output, fake_labels)
        d_loss = d_loss_real + d_loss_fake

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # 生成器损失
        fake_output = discriminator(torch.cat((input_image, fake_target_image), dim=1))
        g_loss = criterion(fake_output, real_labels) + nn.L1Loss()(fake_target_image, target_image)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        g_loss_epoch += g_loss.item()
        d_loss_epoch += d_loss.item()

    # 记录每个 epoch 的平均损失
    g_losses.append(g_loss_epoch / len(dataloader))
    d_losses.append(d_loss_epoch / len(dataloader))

    print(f'Epoch [{epoch+1}/{num_epochs}], d_loss: {d_losses[-1]:.4f}, g_loss: {g_losses[-1]:.4f}')
    generate_images(generator, input_image, target_image)

# 保存生成器模型
torch.save(generator.state_dict(), 'pix2pix_generator_2.pth')

# 绘制损失曲线
plt.figure()
plt.plot(g_losses, label='Generator Loss')
plt.plot(d_losses, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Losses')
plt.show()

# 修改部分2：确保图像转换正确
def load_model_and_generate_images(generator, input_folder, output_folder, device):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

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

            # 将生成的图像从 [-1, 1] 规范化到 [0, 1]
            generated_image = (generated_image + 1) / 2

            # 将生成的图像转换回PIL图像并保存
            generated_image = generated_image.squeeze(0).cpu()
            generated_image = transforms.ToPILImage()(generated_image)
            output_image_path = os.path.join(output_folder, filename)
            generated_image.save(output_image_path)

            # 可选：显示生成的图像
            plt.imshow(generated_image)
            plt.axis('off')
            plt.show()


if __name__ == "__main__":
    input_folder = 'test/input'
    output_folder = 'test/output'
    load_model_and_generate_images(generator, input_folder, output_folder, device)