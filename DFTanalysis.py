import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def compute_fft(image):
    """
    计算图像的傅里叶变换
    返回：幅度谱、相位谱、原始频域数据
    """
    # 如果是彩色图，转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 傅里叶变换
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)  # 将低频移到中心
    
    # 计算幅度谱（取对数增强显示）
    magnitude_spectrum = np.log(np.abs(fshift) + 1)
    
    # 计算相位谱
    phase_spectrum = np.angle(fshift)
    
    return magnitude_spectrum, phase_spectrum, fshift, gray

def show_image(ax, img, title):
    """
    在子图中显示图像，自动处理灰度和彩色图
    """
    if len(img.shape) == 2:  # 灰度图
        ax.imshow(img, cmap='gray')
    else:  # 彩色图
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=12)
    ax.axis('off')

def analyze_frequency_domain(original, gauss_noisy, pepper_noisy, base_name, save_dir):
    """
    对原图和噪声图进行频域分析
    生成完整的频域分析
    """
    # 确保保存目录存在
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 计算各图像的频谱
    orig_magnitude, orig_phase, orig_fshift, orig_gray = compute_fft(original)
    
    gauss_magnitude = gauss_phase = gauss_fshift = gauss_gray = None
    pepper_magnitude = pepper_phase = pepper_fshift = pepper_gray = None
    
    if gauss_noisy is not None:
        gauss_magnitude, gauss_phase, gauss_fshift, gauss_gray = compute_fft(gauss_noisy)
    
    if pepper_noisy is not None:
        pepper_magnitude, pepper_phase, pepper_fshift, pepper_gray = compute_fft(pepper_noisy)
    
    # ========== 空间域与频域综合分析 ==========
    if gauss_noisy is not None and pepper_noisy is not None:
        fig3, axes3 = plt.subplots(3, 3, figsize=(18, 18))
        fig3.suptitle(f'{base_name} - 空间域与频域综合分析', fontsize=14, fontweight='bold')
        
        # 第一行：原图
        show_image(axes3[0, 0], original, '原图 (Spatial Domain)')
        axes3[0, 1].imshow(orig_magnitude, cmap='jet')
        axes3[0, 1].set_title('原图幅度谱', fontsize=12)
        axes3[0, 1].axis('off')
        axes3[0, 2].imshow(orig_phase, cmap='hsv')
        axes3[0, 2].set_title('原图相位谱', fontsize=12)
        axes3[0, 2].axis('off')
        
        # 第二行：高斯噪声
        show_image(axes3[1, 0], gauss_noisy, '高斯噪声 (Spatial Domain)')
        axes3[1, 1].imshow(gauss_magnitude, cmap='jet')
        axes3[1, 1].set_title('高斯噪声幅度谱', fontsize=12)
        axes3[1, 1].axis('off')
        axes3[1, 2].imshow(gauss_phase, cmap='hsv')
        axes3[1, 2].set_title('高斯噪声相位谱', fontsize=12)
        axes3[1, 2].axis('off')
        
        # 第三行：椒盐噪声
        show_image(axes3[2, 0], pepper_noisy, '椒盐噪声 (Spatial Domain)')
        axes3[2, 1].imshow(pepper_magnitude, cmap='jet')
        axes3[2, 1].set_title('椒盐噪声幅度谱', fontsize=12)
        axes3[2, 1].axis('off')
        axes3[2, 2].imshow(pepper_phase, cmap='hsv')
        axes3[2, 2].set_title('椒盐噪声相位谱', fontsize=12)
        axes3[2, 2].axis('off')
        
        plt.tight_layout()
        fig3.savefig(os.path.join(save_dir, f'{base_name}_full_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {base_name}_full_analysis.png")
    
    # ========== 频谱差异分析 ==========
    if gauss_noisy is not None and pepper_noisy is not None:
        fig4, axes4 = plt.subplots(2, 2, figsize=(12, 12))
        fig4.suptitle(f'{base_name} - 频谱差异分析', fontsize=14, fontweight='bold')
        
        # 计算频谱差异
        gauss_diff = np.abs(orig_magnitude - gauss_magnitude)
        pepper_diff = np.abs(orig_magnitude - pepper_magnitude)
        
        # 显示差异
        axes4[0, 0].imshow(orig_magnitude, cmap='jet')
        axes4[0, 0].set_title('原图幅度谱 (Reference)', fontsize=12)
        axes4[0, 0].axis('off')
        
        im1 = axes4[0, 1].imshow(gauss_diff, cmap='hot')
        axes4[0, 1].set_title('高斯噪声 - 幅度谱差异', fontsize=12)
        axes4[0, 1].axis('off')
        plt.colorbar(im1, ax=axes4[0, 1])
        
        axes4[1, 0].imshow(pepper_diff, cmap='hot')
        axes4[1, 0].set_title('椒盐噪声 - 幅度谱差异', fontsize=12)
        axes4[1, 0].axis('off')
        
        # 频谱差异直方图
        axes4[1, 1].hist(gauss_diff.flatten(), bins=50, alpha=0.7, label='高斯噪声', color='blue')
        axes4[1, 1].hist(pepper_diff.flatten(), bins=50, alpha=0.7, label='椒盐噪声', color='red')
        axes4[1, 1].set_title('频谱差异分布直方图', fontsize=12)
        axes4[1, 1].set_xlabel('差异值')
        axes4[1, 1].set_ylabel('频率')
        axes4[1, 1].legend()
        axes4[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig4.savefig(os.path.join(save_dir, f'{base_name}_spectrum_difference.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {base_name}_spectrum_difference.png")
    
    # ========== 中心频率剖面分析 ==========
    if orig_gray is not None:
        fig5, axes5 = plt.subplots(2, 2, figsize=(14, 10))
        fig5.suptitle(f'{base_name} - 频率剖面分析', fontsize=14, fontweight='bold')
        
        # 获取图像中心的行和列
        center_row = orig_gray.shape[0] // 2
        center_col = orig_gray.shape[1] // 2
        
        # 提取中心行和列的频率分布
        orig_row_profile = orig_magnitude[center_row, :]
        
        axes5[0, 0].plot(orig_row_profile)
        axes5[0, 0].set_title('原图幅度谱 - 中心行频率分布', fontsize=12)
        axes5[0, 0].set_xlabel('频率位置')
        axes5[0, 0].set_ylabel('幅度 (log)')
        axes5[0, 0].grid(True, alpha=0.3)
        
        if gauss_magnitude is not None:
            gauss_row_profile = gauss_magnitude[center_row, :]
            axes5[0, 1].plot(orig_row_profile, label='原图', alpha=0.7)
            axes5[0, 1].plot(gauss_row_profile, label='高斯噪声', alpha=0.7)
            axes5[0, 1].set_title('中心行频率对比 (原图 vs 高斯噪声)', fontsize=12)
            axes5[0, 1].set_xlabel('频率位置')
            axes5[0, 1].set_ylabel('幅度 (log)')
            axes5[0, 1].legend()
            axes5[0, 1].grid(True, alpha=0.3)
        
        if pepper_magnitude is not None:
            pepper_row_profile = pepper_magnitude[center_row, :]
            axes5[1, 0].plot(orig_row_profile, label='原图', alpha=0.7)
            axes5[1, 0].plot(pepper_row_profile, label='椒盐噪声', alpha=0.7)
            axes5[1, 0].set_title('中心行频率对比 (原图 vs 椒盐噪声)', fontsize=12)
            axes5[1, 0].set_xlabel('频率位置')
            axes5[1, 0].set_ylabel('幅度 (log)')
            axes5[1, 0].legend()
            axes5[1, 0].grid(True, alpha=0.3)
        
        if gauss_magnitude is not None and pepper_magnitude is not None:
            axes5[1, 1].plot(orig_row_profile, label='原图', alpha=0.7)
            axes5[1, 1].plot(gauss_row_profile, label='高斯噪声', alpha=0.7)
            axes5[1, 1].plot(pepper_row_profile, label='椒盐噪声', alpha=0.7)
            axes5[1, 1].set_title('三种图像中心行频率对比', fontsize=12)
            axes5[1, 1].set_xlabel('频率位置')
            axes5[1, 1].set_ylabel('幅度 (log)')
            axes5[1, 1].legend()
            axes5[1, 1].grid(True, alpha=0.3)
        elif gauss_magnitude is None and pepper_magnitude is not None:
            axes5[1, 1].plot(orig_row_profile, label='原图', alpha=0.7)
            axes5[1, 1].plot(pepper_row_profile, label='椒盐噪声', alpha=0.7)
            axes5[1, 1].set_title('中心行频率对比 (原图 vs 椒盐噪声)', fontsize=12)
            axes5[1, 1].set_xlabel('频率位置')
            axes5[1, 1].set_ylabel('幅度 (log)')
            axes5[1, 1].legend()
            axes5[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig5.savefig(os.path.join(save_dir, f'{base_name}_frequency_profile.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {base_name}_frequency_profile.png")

def process_images(misc_noise_dir='misc-noise', result_dir='result1'):
    """
    处理misc-noise文件夹中的所有图像，进行频域分析
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
    print("="*50)
    
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
            print(f"未找到 {base_name} 的噪声图，跳过")
            continue
        
        print(f"\n正在分析: {base_name}")
        if gauss_noisy is not None:
            print(f"  - 高斯噪声图: {gauss_file}")
        if pepper_noisy is not None:
            print(f"  - 椒盐噪声图: {pepper_file}")
        
        # 进行频域分析
        analyze_frequency_domain(original, gauss_noisy, pepper_noisy, base_name, result_dir)
        print(f"完成 {base_name} 的频域分析")
        print("-"*50)

def main():
    """
    主函数
    """
    print("="*18)
    print("频域分析程序")
    print("="*18)
    print("分析内容：")
    print("1. 空间域与频域综合分析")
    print("2. 频谱差异分析")
    print("3. 频率剖面分析")
    print("="*18)
    
    # 处理图片
    process_images(misc_noise_dir='misc-noise', result_dir='result1')
    
    print(f"\n所有频域分析完成！结果保存在 result1 文件夹中。")
    print("生成的文件包括：")
    print("  - *_full_analysis.png (综合分析)")
    print("  - *_spectrum_difference.png (频谱差异)")
    print("  - *_frequency_profile.png (频率剖面)")

if __name__ == '__main__':
    main()