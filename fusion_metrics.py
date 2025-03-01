'''
Author: Mina Han
Date: 2024-12-26 06:48:07
LastEditTime: 2024-12-31 01:54:05
Description: 
FilePath: /TextFusion-main/fusion_metrics.py
'''
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2
from skimage import io, img_as_float, color
from skimage.metrics import mean_squared_error
from scipy.ndimage import sobel
import os
import pandas as pd
from skimage.transform import resize
from scipy.signal import convolve2d
import math
import sklearn.metrics as skm

# from vif import vifp_mscale  # 使用 vif-py 库，需要提前安装


# 加载图像
def load_image(path):
    """
    加载图像并确保其为灰度图。

    参数:
        path: 图像文件的路径。

    返回:
        灰度图像，值范围为 [0, 1]。
    """
    # 加载图像
    image = img_as_float(io.imread(path))
    
    # 如果图像不是灰度图（多通道），则转换为灰度图
    if image.ndim == 3:  # RGB 或多通道图像
        image = color.rgb2gray(image)
    
    return image

# 获取指定文件夹内的图像文件
def get_image_files(folder, extension):
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    extension = extension.lower()
    if extension not in valid_extensions:
        raise ValueError(f"Unsupported extension: {extension}")
    return sorted(
        [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() == extension]
    )
def resize_image(image, target_shape):
    """
    将图像调整为目标形状
    参数:
        image: 待调整的图像
        target_shape: 目标尺寸 (height, width)
    返回:
        调整后的图像
    """
    return resize(image, target_shape, mode='reflect', anti_aliasing=True)

# 1. MSE
def calculate_mse(reference, fused):
    return mean_squared_error(reference, fused)

# 2. AG
def calculate_ag(img):  # Average gradient
    Gx, Gy = np.zeros_like(img), np.zeros_like(img)

    Gx[:, 0] = img[:, 1] - img[:, 0]
    Gx[:, -1] = img[:, -1] - img[:, -2]
    Gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2

    Gy[0, :] = img[1, :] - img[0, :]
    Gy[-1, :] = img[-1, :] - img[-2, :]
    Gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2
    return np.mean(np.sqrt((Gx ** 2 + Gy ** 2) / 2))

# 3. ssim
def calculate_ssim(image_A, image_F,):
    return ssim(image_F,image_A,data_range=255)

# 4. CC
def calculate_cc(image_A, image_F):
    rAF = np.sum((image_A - np.mean(image_A)) * (image_F - np.mean(image_F))) / np.sqrt(
        (np.sum((image_A - np.mean(image_A)) ** 2)) * (np.sum((image_F - np.mean(image_F)) ** 2)))
    return rAF

def process_images_and_save_to_excel(visible_folder, infrared_folder, fused_folder, output_excel,method_name):
    # 获取文件名列表（假设所有文件夹内的文件名一致）
    # 获取文件名列表（只取文件名，不包括扩展名）
    visible_files = {os.path.splitext(f)[0]: f for f in get_image_files(visible_folder, ".png")}
    infrared_files = {os.path.splitext(f)[0]: f for f in get_image_files(infrared_folder, ".png")}
    fused_files = {os.path.splitext(f)[0]: f for f in get_image_files(fused_folder, ".png")}

    # 确保文件名一致
    common_keys = set(visible_files.keys()) & set(infrared_files.keys()) & set(fused_files.keys())
    if not common_keys:
        raise ValueError("No matching files found across folders.")

    # 初始化结果字典
    results = {
        "MSE": [],
        "AG": [],
        "SSIM": [],
        "CC":[]
    }

    # 遍历每组图像
    for key in sorted(common_keys):  # 按文件名排序
        visible_path = os.path.join(visible_folder, visible_files[key])
        infrared_path = os.path.join(infrared_folder, infrared_files[key])
        fused_path = os.path.join(fused_folder, fused_files[key])

        # 加载图像
        visible_image = load_image(visible_path)
        infrared_image = load_image(infrared_path)
        fused_image = load_image(fused_path)

        target_shape = visible_image.shape  # 以可见光图像为基准
        if fused_image.shape != target_shape:
            print(f"Resizing fused image: {key}.png from {fused_image.shape} to {target_shape}")
            fused_image = resize_image(fused_image, target_shape)

        # 计算指标
        mse_visible = calculate_mse(visible_image, fused_image)
        mse_infrared = calculate_mse(infrared_image, fused_image)
        ag = calculate_ag(fused_image)
        ssim_visible = calculate_ssim(visible_image,fused_image)
        ssim_infrared = calculate_ssim(infrared_image, fused_image)
        cc_visible = calculate_cc(visible_image, fused_image)
        cc_infrared = calculate_cc(infrared_image, fused_image)

        # 保存结果
        results["CC"].append({
            "Image": f"{key}.png",  # 使用融合图像名
            f"{method_name}-Visible": cc_visible,
            f"{method_name}-Infrared": cc_infrared
        })
        results["SSIM"].append({
            "Image": f"{key}.png",  # 使用融合图像名
            f"{method_name}-Visible": ssim_visible,
            f"{method_name}-Infrared": ssim_infrared
        })
        results["MSE"].append({
            "Image": f"{key}.png",  # 使用融合图像名
            f"{method_name}-Visible": mse_visible,
            f"{method_name}-Infrared": mse_infrared
        })
        results["AG"].append({"Image": f"{key}.png", f"{method_name}": ag})

    # 保存到 Excel 中的多个工作表
    with pd.ExcelWriter(output_excel) as writer:
        for key, value in results.items():
            # 转换为 DataFrame
            df = pd.DataFrame(value)

            # 添加平均值
            avg_row = df.mean(numeric_only=True)
            avg_row["Image"] = "Average"
            avg_row_df = pd.DataFrame([avg_row])  # 转换为 DataFrame 格式
            df = pd.concat([df, avg_row_df], ignore_index=True)  # 使用 pd.concat 添加平均值行

            # 写入到工作表
            df.to_excel(writer, sheet_name=key, index=False)
    
    print(f"Results saved to {output_excel}")

if __name__ == "__main__":
    # 文件夹路径
    visible_folder = "" # 源数据文件夹1
    infrared_folder = "" # 源数据文件夹2
    fused_folder = "" # 融合数据文件夹2
    output_excel = "" # 结果保存
    # 自定义融合方法名称
    method_name = ""

    # 调用函数处理图像并保存结果
    process_images_and_save_to_excel(visible_folder, infrared_folder, fused_folder, output_excel,method_name)
