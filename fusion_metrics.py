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

# 2. SD
def calculate_sd(fused):
    return np.std(fused)

# 3. VIF
def calculate_vif_custom(ref, dist): # viff of a pair of pictures
    sigma_nsq = 2
    eps = 1e-10

    num = 0.0
    den = 0.0
    for scale in range(1, 5):

        N = 2 ** (4 - scale + 1) + 1
        sd = N / 5.0

        # Create a Gaussian kernel as MATLAB's
        m, n = [(ss - 1.) / 2. for ss in (N, N)]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sd * sd))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            win = h / sumh

        if scale > 1:
            ref = convolve2d(ref, np.rot90(win, 2), mode='valid')
            dist = convolve2d(dist, np.rot90(win, 2), mode='valid')
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        mu1 = convolve2d(ref, np.rot90(win, 2), mode='valid')
        mu2 = convolve2d(dist, np.rot90(win, 2), mode='valid')
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = convolve2d(ref * ref, np.rot90(win, 2), mode='valid') - mu1_sq
        sigma2_sq = convolve2d(dist * dist, np.rot90(win, 2), mode='valid') - mu2_sq
        sigma12 = convolve2d(ref * dist, np.rot90(win, 2), mode='valid') - mu1_mu2

        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12

        g[sigma1_sq < eps] = 0
        sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
        sigma1_sq[sigma1_sq < eps] = 0

        g[sigma2_sq < eps] = 0
        sv_sq[sigma2_sq < eps] = 0

        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= eps] = eps

        num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

    vifp = num / den

    if np.isnan(vifp):
        return 1.0
    else:
        return vifp
    
# 4. Qabf
def Qabf_getArray(img):
    # Sobel Operator Sobel
    h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
    h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(np.float32)
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)

    SAx = convolve2d(img, h3, mode='same')
    SAy = convolve2d(img, h1, mode='same')
    gA = np.sqrt(np.multiply(SAx, SAx) + np.multiply(SAy, SAy))
    aA = np.zeros_like(img)
    aA[SAx == 0] = math.pi / 2
    aA[SAx != 0]=np.arctan(SAy[SAx != 0] / SAx[SAx != 0])
    return gA, aA

def Qabf_getQabf(aA, gA, aF, gF):
    L = 1
    Tg = 0.9994
    kg = -15
    Dg = 0.5
    Ta = 0.9879
    ka = -22
    Da = 0.8
    GAF,AAF,QgAF,QaAF,QAF = np.zeros_like(aA),np.zeros_like(aA),np.zeros_like(aA),np.zeros_like(aA),np.zeros_like(aA)
    GAF[gA>gF]=gF[gA>gF]/gA[gA>gF]
    GAF[gA == gF] = gF[gA == gF]
    GAF[gA <gF] = gA[gA<gF]/gF[gA<gF]
    AAF = 1 - np.abs(aA - aF) / (math.pi / 2)
    QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))
    QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))
    QAF = QgAF* QaAF
    return QAF
def calculate_qabf_pair(image_A, image_F):
        gA, aA = Qabf_getArray(image_A)
        gF, aF = Qabf_getArray(image_F)
        QAF = Qabf_getQabf(aA, gA, aF, gF)

        # 计算QABF
        deno = np.sum(gA)
        nume = np.sum(np.multiply(QAF, gA))
        return nume / (deno + 1e-10)

# 5. SCD
def calculate_scd_pair(image_A, image_F): # The sum of the correlations of differences
    imgF_A = image_F - image_A
    corr = np.sum((image_A - np.mean(image_A)) * (imgF_A - np.mean(imgF_A))) / np.sqrt(
        (np.sum((image_A - np.mean(image_A)) ** 2)) * (np.sum((imgF_A - np.mean(imgF_A)) ** 2)))
    return corr

# 6. MS-SSIM
def calculate_ms_ssim(fused, reference, levels=5):
    """
    计算多尺度 SSIM（MS-SSIM）。
    """
    if reference.max() > 1.0:
        reference = reference / 255.0
    if fused.max() > 1.0:
        fused = fused / 255.0
    msssim = []

    # 逐层降采样计算 SSIM
    for _ in range(levels):
        # 计算当前尺度的 SSIM
        current_ssim = ssim(fused, reference, data_range=fused.max() - fused.min())
        msssim.append(current_ssim)

        # 下采样图像
        fused = cv2.pyrDown(fused)
        reference = cv2.pyrDown(reference)

    # 综合每个尺度的 SSIM 值
    return np.mean(msssim)

# 7. EPI
def calculate_epi(source_image, fused_image):
    """
    计算边缘信息保留度（EPI）
    """
    if source_image.max() > 1.0:
        source_image = source_image / 255.0
    if fused_image.max() > 1.0:
        fused_image = fused_image / 255.0

    grad_source_x = cv2.Sobel(source_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_source_y = cv2.Sobel(source_image, cv2.CV_64F, 0, 1, ksize=3)
    grad_fused_x = cv2.Sobel(fused_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_fused_y = cv2.Sobel(fused_image, cv2.CV_64F, 0, 1, ksize=3)
    grad_source = np.sqrt(grad_source_x**2 + grad_source_y**2)
    grad_fused = np.sqrt(grad_fused_x**2 + grad_fused_y**2)
    dot_product = np.sum(grad_source * grad_fused)
    source_square_sum = np.sum(grad_source**2)
    fused_square_sum = np.sum(grad_fused**2)
    if source_square_sum == 0 or fused_square_sum == 0:
        return 0.0
    return dot_product / (np.sqrt(source_square_sum) * np.sqrt(fused_square_sum))

# 8. SF
def calculate_sf(img):
    return np.sqrt(np.mean((img[:, 1:] - img[:, :-1]) ** 2) + np.mean((img[1:, :] - img[:-1, :]) ** 2))

# 9. AG
def calculate_ag(img):  # Average gradient
    Gx, Gy = np.zeros_like(img), np.zeros_like(img)

    Gx[:, 0] = img[:, 1] - img[:, 0]
    Gx[:, -1] = img[:, -1] - img[:, -2]
    Gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2

    Gy[0, :] = img[1, :] - img[0, :]
    Gy[-1, :] = img[-1, :] - img[-2, :]
    Gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2
    return np.mean(np.sqrt((Gx ** 2 + Gy ** 2) / 2))

# 10. MI
def calculate_mi(image_A, image_F):
    return skm.mutual_info_score(image_F.flatten(), image_A.flatten())

# 11. EN
def calculate_entropy(img):  # entropy
    a = np.uint8(np.round(img)).flatten()
    h = np.bincount(a) / a.shape[0]
    return -sum(h * np.log2(h + (h == 0)))

# 12. psnr
def calculate_psnr(image1, image2):
    """
    计算峰值信噪比（PSNR）
    """
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

# 13. ssim
def calculate_ssim(image_A, image_F,):
    return ssim(image_F,image_A,data_range=255)

# 14. CC
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
        "SD": [],
        "VIF": [],
        "Qabf": [],
        "SCD": [],
        "MS-SSIM": [],
        "EPI": [], 
        "SF": [], 
        "AG": [], 
        "MI": [], 
        "Entropy": [], 
        "PSNR": [],
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
        sd_visible = calculate_sd(fused_image)
        sd_infrared = calculate_sd(fused_image)
        vif_lib_visible = calculate_vif_custom(visible_image, fused_image)
        vif_lib_infrared = calculate_vif_custom(infrared_image, fused_image)
        qabf_visible = calculate_qabf_pair(fused_image, visible_image)
        qabf_infrared = calculate_qabf_pair(fused_image, infrared_image)
        scd_visible = calculate_scd_pair(fused_image, visible_image)
        scd_infrared = calculate_scd_pair(fused_image, infrared_image)
        ms_ssim_visible = calculate_ms_ssim(fused_image, visible_image, levels=5)
        ms_ssim_infrared = calculate_ms_ssim(fused_image, infrared_image, levels=5)

        epi_visible = calculate_epi(visible_image, fused_image)
        epi_infrared = calculate_epi(infrared_image, fused_image)
        sf = calculate_sf(fused_image)
        ag = calculate_ag(fused_image)
        mi_visible = calculate_mi(visible_image, fused_image)
        mi_infrared = calculate_mi(infrared_image, fused_image)
        entropy = calculate_entropy(fused_image)
        psnr_visible = calculate_psnr(visible_image, fused_image)
        psnr_infrared = calculate_psnr(infrared_image, fused_image)
        ssim_visible = calculate_ssim(visible_image,fused_image)
        ssim_infrared = calculate_ssim(infrared_image, fused_image)
        cc_visible = calculate_cc(visible_image, fused_image)
        cc_infrared = calculate_cc(infrared_image, fused_image)


        # 打印中间结果
        print(f"Processing file: {key}")
        print(f"  MSE (Visible): {mse_visible:.4f}, MSE (Infrared): {mse_infrared:.4f}")
        print(f"  SD (Visible): {sd_visible:.4f}, SD (Infrared): {sd_infrared:.4f}")
        print(f"  VIF (Visible): {vif_lib_visible:.4f}, VIF (Infrared): {vif_lib_infrared:.4f}")
        print(f"  Qabf (Visible): {qabf_visible:.4f}, Qabf (Infrared): {qabf_infrared:.4f}")
        print(f"  SCD (Visible): {scd_visible:.4f}, SCD (Infrared): {scd_infrared:.4f}")
        print(f"  MS-SSIM (Visible): {ms_ssim_visible:.4f}, MS-SSIM (Infrared): {ms_ssim_infrared:.4f}")
        print(f"  EPI (Visible): {epi_visible:.4f}, EPI (Infrared): {epi_infrared:.4f}")
        print(f"  SF: {sf:.4f}, AG: {ag:.4f}")
        print(f"  MI (Visible): {mi_visible:.4f}, MI (Infrared): {mi_infrared:.4f}")
        print(f"  Entropy: {entropy:.4f}")
        print(f"  PSNR (Visible): {psnr_visible:.4f}, PSNR (Infrared): {psnr_infrared:.4f}")
        print(f"  SSIM (Visible): {ssim_visible:.4f}, SSIM (Infrared): {ssim_infrared:.4f}")
        print(f"  CC (Visible): {cc_visible:.4f}, CC (Infrared): {cc_infrared:.4f}")

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
        results["SD"].append({
            "Image": f"{key}.png",
            f"{method_name}-Visible": sd_visible,
            f"{method_name}-Infrared": sd_infrared
        })
        results["VIF"].append({
            "Image": f"{key}.png",
            f"{method_name}-Visible": vif_lib_visible,
            f"{method_name}-Infrared": vif_lib_infrared
        })
        results["Qabf"].append({
            "Image": f"{key}.png",
            f"{method_name}-Visible": qabf_visible,
            f"{method_name}-Infrared": qabf_infrared
        })
        results["SCD"].append({
            "Image": f"{key}.png",
            f"{method_name}-Visible": scd_visible,
            f"{method_name}-Infrared": scd_infrared
        })
        results["MS-SSIM"].append({
            "Image": f"{key}.png",
            f"{method_name}-Visible": ms_ssim_visible,
            f"{method_name}-Infrared": ms_ssim_infrared
        })
        results["EPI"].append({
            "Image": f"{key}.png",
            f"{method_name}-Visible": epi_visible,
            f"{method_name}-Infrared": epi_infrared
        })
        results["SF"].append({"Image": f"{key}.png", f"{method_name}": sf})
        results["AG"].append({"Image": f"{key}.png", f"{method_name}": ag})
        results["MI"].append({
            "Image": f"{key}.png",
            f"{method_name}-Visible": mi_visible,
            f"{method_name}-Infrared": mi_infrared
        })
        results["Entropy"].append({"Image": f"{key}.png", f"{method_name}": entropy})
        results["PSNR"].append({
            "Image": f"{key}.png",
            f"{method_name}-Visible": psnr_visible,
            f"{method_name}-Infrared": psnr_infrared
        })

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