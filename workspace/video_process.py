import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False

only_first_frame = True
white_line = True


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

def read_video(name):
    cap = cv2.VideoCapture(name)
    num = 1
    frames=[]
    while True:
        success, data = cap.read()
        if not success:
            break
        # im = Image.fromarray(data)
        frames.append(data)
        print("正在读取第"+str(num)+"帧")
        num = num + 1
        if only_first_frame:
            break
    cap.release()
    return frames

def show_coordinate(path):
    image = cv2.imread(path)
    image_height, image_width = image.shape[:2]

    # 创建Matplotlib图像
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 添加坐标轴标注信息
    plt.xlim(0, image_width)
    plt.ylim(image_height, 0)
    # plt.gca().invert_yaxis()
    ax = plt.gca()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.xlabel('Y轴')
    plt.ylabel('X轴')
    plt.title('图像坐标系')

    # 显示坐标系
    plt.grid(True)
    plt.show()



def apply_white_balance(frame):
    # 白平衡
    result = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    if False:
        print(avg_a,avg_b)
        exit()
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def dark_channel(image, size=50):
    """计算暗通道"""
    b, g, r = cv2.split(image)
    min_img = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark = cv2.erode(min_img, kernel)
    return dark

def atmospheric_light(image, dark):
    """估计大气光"""
    h, w = image.shape[:2]
    num_pixels = h * w
    num_brightest = int(max(np.floor(num_pixels / 1000), 1))
    dark_vec = dark.reshape(num_pixels)
    image_vec = image.reshape(num_pixels, 3)

    indices = dark_vec.argsort()[-num_brightest:]
    brightest_pixels = image_vec[indices]
    A = brightest_pixels.max(axis=0)
    return A

def transmission_estimate(image, A, omega=0.8, size=15):
    """估计透射率"""
    norm_img = image.astype(np.float32) / A
    transmission = 1 - omega * dark_channel(norm_img, size)
    return transmission

def guided_filter(I, p, r, eps):
    """引导滤波"""
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * I + mean_b
    return q

def transmission_refine(image, et):
    """使用引导滤波优化透射率"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0
    r = 60
    eps = 0.0001
    t = guided_filter(gray, et, r, eps)
    return t

def recover(image, t, A, t0=0.7):
    """恢复去雾图像"""
    res = np.empty_like(image, dtype=np.float32)
    t = cv2.max(t, t0)

    for i in range(3):
        res[:, :, i] = (image[:, :, i] - A[i]) / t + A[i]

    return cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def dehaze(image):
    """去雾主函数"""
    dark = dark_channel(image)
    A = atmospheric_light(image, dark)
    te = transmission_estimate(image, A)
    t = transmission_refine(image, te)
    result = recover(image, t, A)
    return result

def bilateral_filtered(image):
    # d: 邻域直径
    # sigmaColor: 颜色空间的标准差
    # sigmaSpace: 坐标空间的标准差
    bilateral_filtered_image = cv2.bilateralFilter(image, d=3, sigmaColor=75, sigmaSpace=75)
    return bilateral_filtered_image

def plot_histograms(images, titles):
    num_images = len(images)
    plt.figure(figsize=(12, 4 * num_images))

    for i, (image, title) in enumerate(zip(images, titles)):
        # 显示图像
        plt.subplot(num_images, 2, 2 * i + 1)
        plt.title(f"{title} - 图像")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        # 计算并显示直方图
        plt.subplot(num_images, 2, 2 * i + 2)
        plt.title(f"{title} - 直方图")
        color = ('b', 'g', 'r')
        for j, col in enumerate(color):
            hist = cv2.calcHist([image], [j], None, [256], [0, 256])
            plt.plot(hist, color=col)
            plt.xlim([0, 256])
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()




def show(images,titles):
    num_images = len(images)
    plt.figure(figsize=(12 * num_images, 6))

    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1,num_images, i+1)
        plt.title(f"{title} - 图像")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def output_frame(frame,folder,name):
    if white_line:
        print("draw white line")
        # cv2.line(frame, (0, int(-3.5 * 0 + 875)), (image.shape[1], int(-3.5 * image.shape[1] + 875)),(255, 255, 255), thickness=2)
        cv2.line(frame, (0, 240), (350, 200), (255, 255, 255), thickness=2)
    if not os.path.exists(folder):
        os.makedirs(folder)
    im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    im.save(os.path.join(folder, f"{name}.png"))


def frame_to_avi(frames,folder,name,fps=25):
    if not frames:
        print("帧列表为空，无法创建视频。")
        return

        # 获取帧的宽度和高度
    height, width, layers = frames[0].shape
    size = (width, height)

    # 定义编解码器并创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用 XVID 编码
    out = cv2.VideoWriter(os.path.join(folder,f'{name}.avi'), fourcc, fps, size)

    for frame in frames:
        out.write(frame)

    out.release()
    print("视频已保存到"+os.path.join(folder,f'{name}.avi'))



avi_name="第三次人赛A题附件.avi"

# display_avi(avi_name)

frames=read_video(avi_name)
output_frame(frames[0],"show","ord")
show_coordinate("show/ord.png")
# apply_white_balance
for i in range(len(frames)):
    if i % 100 == 0:
        print("正在对第"+str(i+1)+"帧 调整白平衡")
    frames[i]=apply_white_balance(frames[i])
    if only_first_frame:
        break
output_frame(frames[0],"show","white_balance")
plot_histograms(
    [cv2.imread("show/ord.png"),cv2.imread("show/white_balance.png")],
    ["原始", "白平衡调整"]
)
# dehaze
for i in range(len(frames)):
    if i % 100 == 0:
        print("正在对第"+str(i+1)+"帧 去雾")
    frames[i]=dehaze(frames[i])
    if only_first_frame:
        break
output_frame(frames[0],"show","white_balance_and_dehaze")
plot_histograms(
    [cv2.imread("show/white_balance.png"),cv2.imread("show/white_balance_and_dehaze.png")],
    ["白平衡调整", "白平衡调整并去雾"]
)

# bilateral_filtered
print(type(frames[0]))
for i in range(len(frames)):
    if i % 100 == 0:
        print("正在对第"+str(i+1)+"帧 双边滤波")
    frames[i]=bilateral_filtered(frames[i])
    if only_first_frame:
        break
output_frame(frames[0],"show","bilateral_filtered")

plot_histograms(
    [cv2.imread("show/white_balance_and_dehaze.png"),cv2.imread("show/bilateral_filtered.png")],
    ["调整白平衡并去雾", "双边滤波"]
)

# 放大对比1
im0=cv2.imread("show/white_balance_and_dehaze.png")
im1=cv2.imread("show/bilateral_filtered.png")
choose_area=(245,148,280,215)
im0_cropped = im0[choose_area[0]:choose_area[2], choose_area[1]:choose_area[3]]
im1_cropped = im1[choose_area[0]:choose_area[2], choose_area[1]:choose_area[3]]
scale_factor = 4
height, width = im0.shape[:2]
new_width = int(width * scale_factor)
new_height = int(height * scale_factor)
im0_resized = cv2.resize(im0_cropped , (new_width, new_height), interpolation=cv2.INTER_CUBIC)
im1_resized = cv2.resize(im1_cropped , (new_width, new_height), interpolation=cv2.INTER_CUBIC)
show([im0_resized,im1_resized],["双边滤波前","双边滤波后"])
cv2.waitKey(0)
cv2.destroyAllWindows()


# 放大对比2
im0=cv2.imread("show/white_balance_and_dehaze.png")
im1=cv2.imread("show/bilateral_filtered.png")
choose_area=(211,228,238,260)
im0_cropped = im0[choose_area[0]:choose_area[2], choose_area[1]:choose_area[3]]
im1_cropped = im1[choose_area[0]:choose_area[2], choose_area[1]:choose_area[3]]
scale_factor = 4
height, width = im0.shape[:2]
new_width = int(width * scale_factor)
new_height = int(height * scale_factor)
im0_resized = cv2.resize(im0_cropped , (new_width, new_height), interpolation=cv2.INTER_CUBIC)
im1_resized = cv2.resize(im1_cropped , (new_width, new_height), interpolation=cv2.INTER_CUBIC)
show([im0_resized,im1_resized],["双边滤波前","双边滤波后"])
cv2.waitKey(0)
cv2.destroyAllWindows()


# output
print("输出所有图像到“output”")
for i in range(len(frames)):
    # output_frame(frames[i], "output", format(i, "05"))
    if only_first_frame:
        break
print("输出所有图像完毕")

frame_to_avi(frames,"show","video_process")


plot_histograms(
    [cv2.imread("show/ord.png"),cv2.imread("show/white_balance.png"),cv2.imread("show/white_balance_and_dehaze.png"),cv2.imread("show/bilateral_filtered.png")],
    ["原始", "白平衡调整", "白平衡调整并去雾","双边滤波"]
)