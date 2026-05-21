import os
import cv2
import numpy as np

def add_gaussian_noise(img, mean=0, sigma=25):
    """
    给图像添加高斯噪声
    :param img: 输入图像 
    :param mean: 噪声均值
    :param sigma: 噪声标准差
    :return: 添加噪声后的图像 
    """
    # 生成高斯噪声
    noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    noisy_img = img.astype(np.float32) + noise
    # 裁剪到有效范围并转回原始类型
    if img.dtype == np.uint8:
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    elif img.dtype == np.uint16:
        noisy_img = np.clip(noisy_img, 0, 65535).astype(np.uint16)
    else:
        # 其他类型按原范围处理
        noisy_img = noisy_img.astype(img.dtype)
    return noisy_img

def add_salt_pepper_noise(img, prob=0.02):
    """
    给图像添加椒盐噪声
    :param img: 输入图像 
    :param prob: 噪声比例（默认 2%）
    :return: 添加噪声后的图像 
    """
    noisy_img = img.copy()
    # 确定图像的取值范围
    if img.dtype == np.uint8:
        min_val, max_val = 0, 255
    elif img.dtype == np.uint16:
        min_val, max_val = 0, 65535
    else:
        min_val, max_val = np.iinfo(img.dtype).min, np.iinfo(img.dtype).max

    # 椒盐噪声掩码
    salt_mask = np.random.random(img.shape[:2]) < prob / 2
    pepper_mask = np.random.random(img.shape[:2]) < prob / 2

    # 为彩色图像处理所有通道
    if len(img.shape) == 3:
        salt_mask = np.stack([salt_mask]*img.shape[2], axis=2)
        pepper_mask = np.stack([pepper_mask]*img.shape[2], axis=2)

    noisy_img[salt_mask] = max_val    # 盐噪声 (白点)
    noisy_img[pepper_mask] = min_val  # 椒噪声 (黑点)
    return noisy_img

def process_folder(input_folder):
    """
    遍历文件夹中的 tiff 图像，分别添加高斯和椒盐噪声并保存
    """
    valid_extensions = ('.tif', '.tiff')
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(valid_extensions):
            img_path = os.path.join(input_folder, filename)
            # 读取图像
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"无法读取 {filename}，跳过")
                continue

            # 分离文件名和扩展名
            base_name, ext = os.path.splitext(filename)

            # 添加高斯噪声，保存为 "原文件名_gauss.tiff"
            gauss_img = add_gaussian_noise(img)
            gauss_save_path = os.path.join(input_folder, f"{base_name}_gauss.tiff")
            cv2.imwrite(gauss_save_path, gauss_img)
            print(f"已保存：{gauss_save_path}")

            # 添加椒盐噪声，保存为 "原文件名_pepper.tiff"
            sp_img = add_salt_pepper_noise(img)
            sp_save_path = os.path.join(input_folder, f"{base_name}_pepper.tiff")
            cv2.imwrite(sp_save_path, sp_img)
            print(f"已保存：{sp_save_path}")
            
def rename_folder(old_path, new_name):
    """
    重命名文件夹
    :param old_path: 原文件夹路径
    :param new_name: 新文件夹名称
    """
    # 获取原文件夹的父目录
    parent_dir = os.path.dirname(old_path)
    # 构建新文件夹路径
    new_path = os.path.join(parent_dir, new_name)
    
    try:
        os.rename(old_path, new_path)
        print(f"文件夹已重命名：{old_path} -> {new_path}")
        return new_path
    except Exception as e:
        print(f"重命名失败：{e}")
        return old_path


if __name__ == "__main__":
    # 请将下面的路径替换为你的实际文件夹路径
    input_folder = "D:\\signal-and-system-team-work-master\\misc"
    process_folder(input_folder)
    new_folder_name = "misc-noise"
    renamed_folder = rename_folder(input_folder, new_folder_name)