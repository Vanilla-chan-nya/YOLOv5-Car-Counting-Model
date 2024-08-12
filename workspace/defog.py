import cv2
import numpy as np

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

# 读取图像
image_path = "balance1/split_00001.png"  # 替换为实际图像路径
image = cv2.imread(image_path)

# 去雾
dehazed_image = dehaze(image)

# 显示原始图像和去雾后的图像
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Dehazed Image")
plt.imshow(cv2.cvtColor(dehazed_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()
