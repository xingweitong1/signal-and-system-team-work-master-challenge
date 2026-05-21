import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def wiener_filter(image, kernel_size=3, noise_var=500):
    """
    维纳滤波 - 最适合高斯噪声
    自适应滤波器，根据局部方差调整滤波强度
    """
    # 转换为灰度图处理
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        is_color = True
        channels = cv2.split(image)
    else:
        gray = image
        is_color = False
    
    # 估计噪声方差（如果未指定）
    if noise_var is None:
        # 使用拉普拉斯算子估计噪声
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_var = np.var(laplacian) / np.sqrt(2)
        noise_var = max(noise_var, 1e-10)  # 避免为零
    
    if is_color:
        filtered_channels = []
        for channel in channels:
            # 对每个通道应用维纳滤波
            local_mean = cv2.blur(channel, (kernel_size, kernel_size))
            local_var = cv2.blur(channel.astype(np.float32)**2, (kernel_size, kernel_size)) - local_mean.astype(np.float32)**2
            
            # 维纳滤波公式
            channel_float = channel.astype(np.float32)
            filtered = local_mean.astype(np.float32) + np.maximum(local_var - noise_var, 0) / np.maximum(local_var, noise_var) * (channel_float - local_mean.astype(np.float32))
            
            filtered = np.clip(filtered, 0, 255).astype(np.uint8)
            filtered_channels.append(filtered)
        
        return cv2.merge(filtered_channels)
    else:
        local_mean = cv2.blur(gray, (kernel_size, kernel_size))
        local_var = cv2.blur(gray.astype(np.float32)**2, (kernel_size, kernel_size)) - local_mean.astype(np.float32)**2
        
        gray_float = gray.astype(np.float32)
        filtered = local_mean.astype(np.float32) + np.maximum(local_var - noise_var, 0) / np.maximum(local_var, noise_var) * (gray_float - local_mean.astype(np.float32))
        
        return np.clip(filtered, 0, 255).astype(np.uint8)



def create_butterworth_lowpass_filter(shape, cutoff=30, order=1):
    """
    创建巴特沃斯低通滤波器 - 更平滑的截止特性
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    
    mask = np.zeros((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i - crow)**2 + (j - ccol)**2)
            mask[i, j] = 1 / (1 + (d / cutoff)**(2 * order))
    
    return mask

def apply_frequency_filter(image, mask):
    """
    在频域应用滤波器
    """
    if len(image.shape) == 3:
        channels = cv2.split(image)
        filtered_channels = []
        
        for channel in channels:
            f = np.fft.fft2(channel)
            fshift = np.fft.fftshift(f)
            fshift_filtered = fshift * mask
            f_ishift = np.fft.ifftshift(fshift_filtered)
            img_filtered = np.fft.ifft2(f_ishift)
            img_filtered = np.abs(img_filtered)
            img_filtered = cv2.normalize(img_filtered, None, 0, 255, cv2.NORM_MINMAX)
            filtered_channels.append(np.uint8(img_filtered))
        
        return cv2.merge(filtered_channels)
    else:
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        fshift_filtered = fshift * mask
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_filtered = np.fft.ifft2(f_ishift)
        img_filtered = np.abs(img_filtered)
        img_filtered = cv2.normalize(img_filtered, None, 0, 255, cv2.NORM_MINMAX)
        return np.uint8(img_filtered)

def show_image(ax, img, title):
    """显示图像"""
    if len(img.shape) == 2:
        ax.imshow(img, cmap='gray')
    else:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=11)
    ax.axis('off')

def compute_psnr(original, filtered):
    """计算PSNR"""
    mse = np.mean((original.astype(np.float32) - filtered.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def compute_ssim(original, filtered):
    """计算SSIM（简化版）"""
    # 转换为灰度图
    if len(original.shape) == 3:
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        orig_gray = original
    
    if len(filtered.shape) == 3:
        filt_gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    else:
        filt_gray = filtered
    
    # 计算SSIM相关参数
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    orig_gray = orig_gray.astype(np.float64)
    filt_gray = filt_gray.astype(np.float64)
    
    # 计算均值
    mu1 = cv2.GaussianBlur(orig_gray, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(filt_gray, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # 计算方差和协方差
    sigma1_sq = cv2.GaussianBlur(orig_gray ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(filt_gray ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(orig_gray * filt_gray, (11, 11), 1.5) - mu1_mu2
    
    # SSIM公式
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return np.mean(ssim_map)

def create_analysis_figure(original, gauss_noisy, pepper_noisy,
                          gauss_wiener, pepper_lowpass,
                          gauss_spectrum_before, gauss_spectrum_after,
                          pepper_spectrum_before, pepper_spectrum_after,
                          base_name, save_dir):
    """
    创建综合分析图
    """
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f'{base_name} - 基于频域特性的滤波', fontsize=14, fontweight='bold')
    
    # 布局：3行4列
    # 第一行：原图、高斯噪声、高斯滤波结果、高斯频谱对比
    ax1 = plt.subplot(3, 4, 1)
    ax2 = plt.subplot(3, 4, 2)
    ax3 = plt.subplot(3, 4, 3)
    ax4 = plt.subplot(3, 4, 4)
    
    # 第二行：原图、椒盐噪声、椒盐滤波结果、椒盐频谱对比
    ax5 = plt.subplot(3, 4, 5)
    ax6 = plt.subplot(3, 4, 6)
    ax7 = plt.subplot(3, 4, 7)
    ax8 = plt.subplot(3, 4, 8)
    
    # 第三行：质量指标对比
    ax9 = plt.subplot(3, 1, 3)
    
    # 第一行：高斯噪声处理
    show_image(ax1, original, '原图 (Original)')
    show_image(ax2, gauss_noisy, f'高斯噪声图\nPSNR: {compute_psnr(original, gauss_noisy):.2f}dB')
    show_image(ax3, gauss_wiener, f'维纳滤波结果\n(适合全频段噪声)\nPSNR: {compute_psnr(original, gauss_wiener):.2f}dB')
    ax4.imshow(np.hstack([gauss_spectrum_before, gauss_spectrum_after]), cmap='jet')
    ax4.set_title('频谱对比\n(左:滤波前 右:滤波后)', fontsize=11)
    ax4.axis('off')
    
    # 第二行：椒盐噪声处理
    show_image(ax5, original, '原图 (Original)')
    show_image(ax6, pepper_noisy, f'椒盐噪声图\nPSNR: {compute_psnr(original, pepper_noisy):.2f}dB')
    show_image(ax7, pepper_lowpass, f'低通滤波结果\n(去除高频噪声)\nPSNR: {compute_psnr(original, pepper_lowpass):.2f}dB')
    ax8.imshow(np.hstack([pepper_spectrum_before, pepper_spectrum_after]), cmap='jet')
    ax8.set_title('频谱对比\n(左:滤波前 右:滤波后)', fontsize=11)
    ax8.axis('off')
    
    # 第三行：质量指标对比柱状图
    noise_types = ['高斯噪声', '椒盐噪声']
    psnr_before = [compute_psnr(original, gauss_noisy), compute_psnr(original, pepper_noisy)]
    psnr_after = [compute_psnr(original, gauss_wiener), compute_psnr(original, pepper_lowpass)]
    ssim_before = [compute_ssim(original, gauss_noisy), compute_ssim(original, pepper_noisy)]
    ssim_after = [compute_ssim(original, gauss_wiener), compute_ssim(original, pepper_lowpass)]
    
    x = np.arange(len(noise_types))
    width = 0.2
    
    ax9.bar(x - width, psnr_before, width, label='滤波前PSNR', color='red', alpha=0.7)
    ax9.bar(x, psnr_after, width, label='滤波后PSNR', color='green', alpha=0.7)
    ax9.bar(x + width, ssim_before, width, label='滤波前SSIM', color='orange', alpha=0.7)
    ax9.bar(x + 2*width, ssim_after, width, label='滤波后SSIM', color='blue', alpha=0.7)
    
    ax9.set_xlabel('噪声类型')
    ax9.set_ylabel('质量指标值')
    ax9.set_title('滤波效果量化评估')
    ax9.set_xticks(x)
    ax9.set_xticklabels(noise_types)
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{base_name}_adaptive_filtering.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存综合分析图: {save_path}")



def process_images(misc_noise_dir='misc-noise', result_dir='result3'):
    """
    基于频域特征的滤波
    """
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取原图列表
    tiff_files = [f for f in os.listdir(misc_noise_dir) if f.endswith('.tiff')]
    original_files = [f for f in tiff_files if '_gauss' not in f and '_pepper' not in f]
    
    if not original_files:
        print("未找到原图文件")
        return
    
    print(f"找到 {len(original_files)} 个原图文件")
    print("="*60)
    print("滤波策略：")
    print("- 高斯噪声 → 维纳滤波（自适应全频段处理）")
    print("- 椒盐噪声 → 巴特沃斯低通滤波（针对高频噪声）")
    print("="*60)
    
    for orig_file in original_files:
        orig_path = os.path.join(misc_noise_dir, orig_file)
        base_name = orig_file.replace('.tiff', '')
        
        # 读取原图
        original = cv2.imread(orig_path)
        if original is None:
            print(f"无法读取原图: {orig_path}")
            continue
        
        # 读取噪声图
        gauss_path = os.path.join(misc_noise_dir, f"{base_name}_gauss.tiff")
        pepper_path = os.path.join(misc_noise_dir, f"{base_name}_pepper.tiff")
        
        gauss_noisy = cv2.imread(gauss_path) if os.path.exists(gauss_path) else None
        pepper_noisy = cv2.imread(pepper_path) if os.path.exists(pepper_path) else None
        
        if gauss_noisy is None and pepper_noisy is None:
            print(f"未找到 {base_name} 的噪声图，跳过")
            continue
        
        print(f"\n正在处理: {base_name}")
        
        gauss_wiener = None
        pepper_lowpass = None
        gauss_spectrum_before = gauss_spectrum_after = None
        pepper_spectrum_before = pepper_spectrum_after = None
        
        # 处理高斯噪声 - 使用维纳滤波
        if gauss_noisy is not None:
            print("  → 高斯噪声：应用维纳滤波...")
            gauss_wiener = wiener_filter(gauss_noisy, kernel_size=5)
            
            # 计算频谱用于展示
            gauss_gray = cv2.cvtColor(gauss_noisy, cv2.COLOR_BGR2GRAY) if len(gauss_noisy.shape) == 3 else gauss_noisy
            gauss_filtered_gray = cv2.cvtColor(gauss_wiener, cv2.COLOR_BGR2GRAY) if len(gauss_wiener.shape) == 3 else gauss_wiener
            
            f_before = np.fft.fft2(gauss_gray)
            f_after = np.fft.fft2(gauss_filtered_gray)
            
            gauss_spectrum_before = np.log(np.abs(np.fft.fftshift(f_before)) + 1)
            gauss_spectrum_after = np.log(np.abs(np.fft.fftshift(f_after)) + 1)
            
            # 归一化到相同范围
            vmin = min(gauss_spectrum_before.min(), gauss_spectrum_after.min())
            vmax = max(gauss_spectrum_before.max(), gauss_spectrum_after.max())
            gauss_spectrum_before = (gauss_spectrum_before - vmin) / (vmax - vmin)
            gauss_spectrum_after = (gauss_spectrum_after - vmin) / (vmax - vmin)
        
        # 处理椒盐噪声 - 使用巴特沃斯低通滤波
        if pepper_noisy is not None:
            print("  → 椒盐噪声：应用巴特沃斯低通滤波...")
            # 创建巴特沃斯低通滤波器
            butterworth_mask = create_butterworth_lowpass_filter(pepper_noisy.shape[:2], cutoff=40, order=3)
            pepper_lowpass = apply_frequency_filter(pepper_noisy, butterworth_mask)
            
            # 计算频谱用于展示
            pepper_gray = cv2.cvtColor(pepper_noisy, cv2.COLOR_BGR2GRAY) if len(pepper_noisy.shape) == 3 else pepper_noisy
            pepper_filtered_gray = cv2.cvtColor(pepper_lowpass, cv2.COLOR_BGR2GRAY) if len(pepper_lowpass.shape) == 3 else pepper_lowpass
            
            f_before = np.fft.fft2(pepper_gray)
            f_after = np.fft.fft2(pepper_filtered_gray)
            
            pepper_spectrum_before = np.log(np.abs(np.fft.fftshift(f_before)) + 1)
            pepper_spectrum_after = np.log(np.abs(np.fft.fftshift(f_after)) + 1)
            
            # 归一化
            vmin = min(pepper_spectrum_before.min(), pepper_spectrum_after.min())
            vmax = max(pepper_spectrum_before.max(), pepper_spectrum_after.max())
            pepper_spectrum_before = (pepper_spectrum_before - vmin) / (vmax - vmin)
            pepper_spectrum_after = (pepper_spectrum_after - vmin) / (vmax - vmin)
        
        # 计算PSNR
        gauss_psnr_before = compute_psnr(original, gauss_noisy) if gauss_noisy is not None else 0
        gauss_psnr_after = compute_psnr(original, gauss_wiener) if gauss_wiener is not None else 0
        pepper_psnr_before = compute_psnr(original, pepper_noisy) if pepper_noisy is not None else 0
        pepper_psnr_after = compute_psnr(original, pepper_lowpass) if pepper_lowpass is not None else 0
        
        # 输出PSNR改善结果
        if gauss_noisy is not None:
            print(f"    高斯噪声 PSNR: {gauss_psnr_before:.2f} dB → {gauss_psnr_after:.2f} dB (改善: +{gauss_psnr_after-gauss_psnr_before:.2f} dB)")
        if pepper_noisy is not None:
            print(f"    椒盐噪声 PSNR: {pepper_psnr_before:.2f} dB → {pepper_psnr_after:.2f} dB (改善: +{pepper_psnr_after-pepper_psnr_before:.2f} dB)")
        
        # 创建综合分析图
        if gauss_noisy is not None and pepper_noisy is not None:
            create_analysis_figure(original, gauss_noisy, pepper_noisy,
                                 gauss_wiener, pepper_lowpass,
                                 gauss_spectrum_before, gauss_spectrum_after,
                                 pepper_spectrum_before, pepper_spectrum_after,
                                 base_name, result_dir)
            
 
        
        print(f"完成 {base_name} 的处理")
        print("-"*60)

def main():
    print("="*50)
    print("基于频域特征的滤波")
    print("="*50)
    print("\n滤波策略原理：")
    print("1. 高斯噪声（全频段均匀）→ 维纳滤波")
    print("   - 自适应局部方差估计")
    print("   - 在平坦区域强滤波，边缘区域弱滤波")
    print("   - 保留图像细节的同时去除噪声")
    print("\n2. 椒盐噪声（高频为主）→ 巴特沃斯低通滤波")
    print("   - 针对高频噪声设计")
    print("   - 平滑过渡带，避免振铃效应")
    print("   - 保留低频信号完整性")
    print("="*50)
    
    process_images(misc_noise_dir='misc-noise', result_dir='result3')
    
    print("\n所有处理完成！结果保存在 result3 文件夹中。")
    print("\n生成的文件包括：")
    print("  - *_adaptive_filtering.png (自适应滤波综合分析)")
  

if __name__ == '__main__':
    main()