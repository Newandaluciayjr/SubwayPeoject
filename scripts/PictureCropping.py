from PIL import Image
import os

# 指定文件夹路径
folder_path = 'test/prepare'
output_folder = 'test/input'


# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

jpg_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

# 分割每个PNG文件
for file in jpg_files:

    image = Image.open(os.path.join(folder_path, file))

    # 裁剪原始图像
    width, height = image.size
    print(width, height)

    left = 0
    top = 0
    right = width
    bottom = width / 3

    image = image.crop((left, top, right, bottom))

    image.show()

    segment_width = width // 12
    offset_width = segment_width // 8

    # 分割并保存每个部分
    for i in range(12):
        for n in range(4):
            left_x = i * segment_width + offset_width
            left_y = n * segment_width + offset_width
            right_x = left_x + segment_width - 2 * offset_width
            right_y = left_y + segment_width - 2 * offset_width

            box = (left_x, left_y, right_x, right_y)

            segment = image.crop(box)

            resized_segment = segment.resize((256, 256))

            new_file_name = f"{os.path.splitext(file)[0]}_part{n + 1, i + 1}.png"

            resized_segment.save(os.path.join(output_folder, new_file_name))

print("finished")


