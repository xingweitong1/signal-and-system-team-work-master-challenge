import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def mean_filter(image, kernel_size=3):
    """
    均值滤波
    """
    return cv2.blur(image, (kernel_size, kernel_size))

def median_filter(image, kernel_size=3):
    """
    中值滤波
    """
    return cv2.medianBlur(image, kernel_size)

def show_image(ax, img, title):
    """
    在子图中显示图像，自动处理灰度和彩色图
    """
    if len(img.shape) == 2:  # 灰度图
        ax.imshow(img, cmap='gray')
    else:  # 彩色图
        # OpenCV读入的是BGR，转换为RGB显示
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=12)
    ax.axis('off')

def create_comparison_figure(original, gauss_noisy, pepper_noisy, 
                             gauss_mean, gauss_median, 
                             pepper_mean, pepper_median,
                             base_name, save_path):
    """
    创建对比图：7个子图，分为3行
    第一行：原图、高斯噪声、椒盐噪声
    第二行：高斯均值滤波、高斯中值滤波
    第三行：椒盐均值滤波、椒盐中值滤波
    """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'{base_name} - 滤波效果对比', fontsize=16, fontweight='bold')
    
    
    # 第一行：3个子图
    ax1 = plt.subplot(3, 3, (1, 3))  # 原图居中在第一行
    ax2 = plt.subplot(3, 3, 4)       # 高斯噪声
    ax3 = plt.subplot(3, 3, 5)       # 椒盐噪声
    
    # 第二行：2个子图
    ax4 = plt.subplot(3, 3, (7, 8))  # 高斯均值滤波
    
    # 第三行：2个子图
    ax5 = plt.subplot(3, 4, (9, 11))  # 高斯中值滤波
    ax6 = plt.subplot(3, 4, (14, 16)) # 椒盐均值滤波
    ax7 = plt.subplot(3, 4, (17, 19)) # 椒盐中值滤波
    
    # 显示图像
    show_image(ax1, original, '原图 (Original)')
    show_image(ax2, gauss_noisy, '高斯噪声 (Gauss Noise)')
    show_image(ax3, pepper_noisy, '椒盐噪声 (Pepper Noise)')
    show_image(ax4, gauss_mean, '高斯图-均值滤波 (Gauss Mean)')
    show_image(ax5, gauss_median, '高斯图-中值滤波 (Gauss Median)')
    show_image(ax6, pepper_mean, '椒盐图-均值滤波 (Pepper Mean)')
    show_image(ax7, pepper_median, '椒盐图-中值滤波 (Pepper Median)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存对比图: {save_path}")

def create_comparison_figure_alternative(original, gauss_noisy, pepper_noisy, 
                                         gauss_mean, gauss_median, 
                                         pepper_mean, pepper_median,
                                         base_name, save_path):
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f'{base_name} - 滤波效果对比', fontsize=16, fontweight='bold')
    
    # 隐藏第3列的第2、3行子图
    axes[1, 2].axis('off')
    axes[2, 2].axis('off')
    
    # 第一行：原图、高斯噪声、椒盐噪声
    show_image(axes[0, 0], original, '原图 (Original)')
    show_image(axes[0, 1], gauss_noisy, '高斯噪声 (Gauss Noise)')
    show_image(axes[0, 2], pepper_noisy, '椒盐噪声 (Pepper Noise)')
    
    # 第二行：高斯均值滤波、高斯中值滤波
    show_image(axes[1, 0], gauss_mean, '高斯图-均值滤波 (Gauss Mean)')
    show_image(axes[1, 1], gauss_median, '高斯图-中值滤波 (Gauss Median)')
    
    # 第三行：椒盐均值滤波、椒盐中值滤波
    show_image(axes[2, 0], pepper_mean, '椒盐图-均值滤波 (Pepper Mean)')
    show_image(axes[2, 1], pepper_median, '椒盐图-中值滤波 (Pepper Median)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存对比图: {save_path}")

def process_images(misc_noise_dir='misc-noise', result_dir='result'):
    """
    处理misc-noise文件夹中的所有噪声图片
    """
    # 创建结果文件夹
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取原图列表
    tiff_files = [f for f in os.listdir(misc_noise_dir) if f.endswith('.tiff')]
    
    original_files = []
    for f in tiff_files:
        if '_gauss' not in f and '_pepper' not in f:
            original_files.append(f)
    
    if not original_files:
        print("未找到原图文件，请检查misc-noise文件夹")
        return
    
    print(f"找到 {len(original_files)} 个原图文件")
    
    for orig_file in original_files:
        orig_path = os.path.join(misc_noise_dir, orig_file)
        
        # 生成对应的噪声文件名
        base_name = orig_file.replace('.tiff', '')
        gauss_file = f"{base_name}_gauss.tiff"
        pepper_file = f"{base_name}_pepper.tiff"
        
        gauss_path = os.path.join(misc_noise_dir, gauss_file)
        pepper_path = os.path.join(misc_noise_dir, pepper_file)
        
        # 读取原图
        original = cv2.imread(orig_path)
        if original is None:
            print(f"无法读取原图: {orig_path}")
            continue
        
        # 读取噪声图
        gauss_noisy = cv2.imread(gauss_path) if os.path.exists(gauss_path) else None
        pepper_noisy = cv2.imread(pepper_path) if os.path.exists(pepper_path) else None
        
        if gauss_noisy is None and pepper_noisy is None:
            print(f"未找到 {base_name} 的噪声图")
            continue
        
        print(f"\n处理 {base_name}...")
        
        # 对噪声图进行滤波
        gauss_mean = mean_filter(gauss_noisy) if gauss_noisy is not None else None
        gauss_median = median_filter(gauss_noisy) if gauss_noisy is not None else None
        pepper_mean = mean_filter(pepper_noisy) if pepper_noisy is not None else None
        pepper_median = median_filter(pepper_noisy) if pepper_noisy is not None else None
        
        # 创建对比图
        save_path = os.path.join(result_dir, f"{base_name}_comparison.png")
        
        # 显示完整的7个子图
        if gauss_noisy is not None and pepper_noisy is not None:
            create_comparison_figure_alternative(
                original, gauss_noisy, pepper_noisy,
                gauss_mean, gauss_median,
                pepper_mean, pepper_median,
                base_name, save_path
            )
        
        
        
        print(f"已保存对比图: {save_path}")

def main():
    """
    主函数
    """
    print("开始处理图片...")
    
    
    # 处理图片
    process_images(misc_noise_dir='misc-noise', result_dir='result')
    
    print("\n所有图片处理完成！结果保存在result文件夹中。")

if __name__ == '__main__':
    main()