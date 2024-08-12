import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False

def display_avi(name):
    print("按esc退出播放")
    cap = cv2.VideoCapture(name)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow("frame", frame)
        else:
            print("视频播放完成！")
            break

        key = cv2.waitKey(25)
        if key == 27:  # 按esc退出
            break

    cap.release()
    cv2.destroyAllWindows()

def split_avi(name,folder,skip=0):
    if not os.path.exists(folder):
        os.makedirs(folder)
    cap = cv2.VideoCapture(name)
    num = 1
    while True:
        success, data = cap.read()
        if not success:
            break
        im = Image.fromarray(data)
        im.save(folder+'/split_' + format(num, "05") + ".png")
        print(num)
        num = num + 1
    cap.release()

def apply_white_balance(frame):
    # 白平衡
    result = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def split_avi_balance1(name,folder,skip=0):
    if not os.path.exists(folder):
        os.makedirs(folder)
    cap = cv2.VideoCapture(name)
    num = 1
    while True:
        success, data = cap.read()
        if not success:
            break
        balanced_frame = apply_white_balance(data)


        # 转换为 PIL 图像并保存
        im = Image.fromarray(cv2.cvtColor(balanced_frame, cv2.COLOR_BGR2RGB))
        im.save(folder+'/split_' + format(num, "05") + ".png")
        print(num)
        num = num + 1
    cap.release()


def split_avi_balance2(name,folder,skip=0):
    if not os.path.exists(folder):
        os.makedirs(folder)
    cap = cv2.VideoCapture(name)
    num = 1
    while True:
        success, data = cap.read()
        if not success:
            break
        balanced_frame = apply_white_balance(data)
        # 切割
        # cropped_image = balanced_frame[144:,:]

        # defog_image=dehaze(balanced_frame)
        defog_image = balanced_frame
        # 应用直方图均衡化
        equalized_frame = apply_histogram_equalization(defog_image)

        # 转换为 PIL 图像并保存
        im = Image.fromarray(cv2.cvtColor(equalized_frame, cv2.COLOR_BGR2RGB))
        im.save(folder+'/split_' + format(num, "05") + ".png")
        print(num)
        num = num + 1
    cap.release()

def plot_histograms(images, titles):
    num_images = len(images)
    plt.figure(figsize=(12, 4 * num_images))

    for i, (image, title) in enumerate(zip(images, titles)):
        # 显示图像
        plt.subplot(num_images, 2, 2 * i + 1)
        plt.title(f"{title} - Image")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        # 计算并显示直方图
        plt.subplot(num_images, 2, 2 * i + 2)
        plt.title(f"{title} - Histogram")
        color = ('b', 'g', 'r')
        for j, col in enumerate(color):
            hist = cv2.calcHist([image], [j], None, [256], [0, 256])
            plt.plot(hist, color=col)
            plt.xlim([0, 256])
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


avi_name="第三次人赛A题附件.avi"

display_avi(avi_name)



# split_avi(avi_name,'split')
# split_avi_balance1(avi_name,'balance1')
split_avi_balance2(avi_name,'balance2')

# draw_histograms("split/split_00001.png","原始图像")
# draw_histograms("balance1/split_00001.png","白平衡调整图像")
# draw_histograms("balance2/split_00001.png","白平衡调整和直方图均衡图像")

plot_histograms(
    [cv2.imread("split/split_00001.png"),cv2.imread("balance1/split_00001.png"),cv2.imread("balance2/split_00001.png")],
    ["原始图像", "白平衡调整图像", "白平衡调整和直方图均衡图像"]
)


def super_resolve_images(input_folder, output_folder, model_path, scale=2):
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 初始化超分辨率模型
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("fsrcnn", scale)

    # 遍历输入文件夹中的所有 PNG 图像
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 读取图像
            image = cv2.imread(input_path)

            # 应用超分辨率
            result = sr.upsample(image)

            # 保存结果图像
            cv2.imwrite(output_path, result)
            print(f"Processed {filename}")

# super_resolve_images('balance1', 'FSRCNN3x', 'FSRCNN_x3.pb')


# 读取图像
image_path = "balance2/split_00001.png"  # 替换为实际图像路径
image = cv2.imread(image_path)

# 应用双边滤波
# d: 邻域直径
# sigmaColor: 颜色空间的标准差
# sigmaSpace: 坐标空间的标准差
bilateral_filtered_image = cv2.bilateralFilter(image, d=5, sigmaColor=15, sigmaSpace=25)

# 显示原始图像和滤波后的图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Bilateral Filtered Image")
plt.imshow(cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()