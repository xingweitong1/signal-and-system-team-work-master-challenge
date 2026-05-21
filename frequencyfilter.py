import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_ideal_filter(shape, filter_type, cutoff, width=20):
    """
    创建理想滤波器
    filter_type: 'lowpass', 'highpass', 'bandpass', 'bandreject'
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    
    mask = np.zeros((rows, cols), np.float32)
    
    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i - crow)**2 + (j - ccol)**2)  # 到中心的距离
            
            if filter_type == 'lowpass':
                if d <= cutoff:
                    mask[i, j] = 1
                    
            elif filter_type == 'highpass':
                if d >= cutoff:
                    mask[i, j] = 1
                    
            elif filter_type == 'bandpass':
                if cutoff - width/2 <= d <= cutoff + width/2:
                    mask[i, j] = 1
                    
            elif filter_type == 'bandreject':
                if d < cutoff - width/2 or d > cutoff + width/2:
                    mask[i, j] = 1
    
    return mask

def create_gaussian_filter(shape, filter_type, cutoff, width=20):
    """
    创建高斯滤波器
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    
    mask = np.zeros((rows, cols), np.float32)
    
    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i - crow)**2 + (j - ccol)**2)
            
            if filter_type == 'lowpass':
                mask[i, j] = np.exp(-(d**2) / (2 * (cutoff**2)))
                
            elif filter_type == 'highpass':
                mask[i, j] = 1 - np.exp(-(d**2) / (2 * (cutoff**2)))
                
            elif filter_type == 'bandpass':
                mask[i, j] = np.exp(-((d**2 - cutoff**2)**2) / (2 * (d * width)**2 + 1e-10))
                
            elif filter_type == 'bandreject':
                mask[i, j] = 1 - np.exp(-((d**2 - cutoff**2)**2) / (2 * (d * width)**2 + 1e-10))
    
    return mask

def apply_frequency_filter(image, mask):
    """
    在频域中应用滤波器
    """
    # 如果是彩色图，分别处理每个通道
    if len(image.shape) == 3:
        channels = cv2.split(image)
        filtered_channels = []
        
        for channel in channels:
            # 傅里叶变换
            f = np.fft.fft2(channel)
            fshift = np.fft.fftshift(f)
            
            # 应用滤波器
            fshift_filtered = fshift * mask
            
            # 逆傅里叶变换
            f_ishift = np.fft.ifftshift(fshift_filtered)
            img_filtered = np.fft.ifft2(f_ishift)
            img_filtered = np.abs(img_filtered)
            
            # 归一化到0-255
            img_filtered = cv2.normalize(img_filtered, None, 0, 255, cv2.NORM_MINMAX)
            img_filtered = np.uint8(img_filtered)
            
            filtered_channels.append(img_filtered)
        
        return cv2.merge(filtered_channels)
    
    else:
        # 灰度图
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

def create_comprehensive_comparison(original, noisy_list, filtered_results, noise_types, 
                                   filter_types, base_name, save_dir):
    """
    创建综合对比图
    """
    num_noises = len(noisy_list)
    num_filters = len(filter_types)
    
    # 计算总行数：原图+每种噪声的结果
    total_rows = 1 + num_noises
    
    fig, axes = plt.subplots(total_rows, 1 + num_filters, figsize=(4*(1+num_filters), 4*total_rows))
    
    # 如果只有一行，需要reshape
    if total_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'{base_name} - 频域滤波效果对比', fontsize=14, fontweight='bold')
    
    # 第一行：原图和原图的滤波结果
    show_image(axes[0, 0], original, '原图 (Original)')
    axes[0, 1].axis('off')
    axes[0, 2].axis('off')
    axes[0, 3].axis('off')
    
    
    # 后续行：噪声图和滤波结果
    for i, (noisy, noise_type) in enumerate(zip(noisy_list, noise_types)):
        show_image(axes[i+1, 0], noisy, f'{noise_type}噪声图')
        for j, filter_type in enumerate(filter_types):
            if noise_type in filtered_results:
                show_image(axes[i+1, j+1], filtered_results[noise_type][j], 
                          f'{noise_type}-{filter_type}')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{base_name}_frequency_filter_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {save_path}")



def create_spectrum_comparison(original, noisy_list, filtered_results, noise_types, 
                              filter_types, base_name, save_dir):
    """
    创建频域滤波前后的频谱对比
    """
    num_noises = len(noisy_list)
    
    # 对每种噪声类型创建频谱对比
    for i, (noisy, noise_type) in enumerate(zip(noisy_list, noise_types)):
        fig, axes = plt.subplots(2, len(filter_types) + 1, figsize=(4*(len(filter_types)+1), 8))
        
        if len(filter_types) == 0:
            continue
            
        fig.suptitle(f'{base_name} - {noise_type}噪声频域滤波频谱对比', fontsize=14, fontweight='bold')
        
        # 计算原始噪声图的频谱
        noisy_gray = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY) if len(noisy.shape) == 3 else noisy
        f_noisy = np.fft.fft2(noisy_gray)
        fshift_noisy = np.fft.fftshift(f_noisy)
        magnitude_noisy = np.log(np.abs(fshift_noisy) + 1)
        
        # 显示噪声图及其频谱
        show_image(axes[0, 0], noisy, f'{noise_type}噪声图')
        axes[1, 0].imshow(magnitude_noisy, cmap='jet')
        axes[1, 0].set_title('滤波前幅度谱', fontsize=11)
        axes[1, 0].axis('off')
        
        # 显示滤波后的图像和频谱
        for j, filter_type in enumerate(filter_types):
            if noise_type in filtered_results:
                filtered_img = filtered_results[noise_type][j]
                
                # 计算滤波后图像的频谱
                filtered_gray = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY) if len(filtered_img.shape) == 3 else filtered_img
                f_filtered = np.fft.fft2(filtered_gray)
                fshift_filtered = np.fft.fftshift(f_filtered)
                magnitude_filtered = np.log(np.abs(fshift_filtered) + 1)
                
                show_image(axes[0, j+1], filtered_img, f'{filter_type}滤波结果')
                axes[1, j+1].imshow(magnitude_filtered, cmap='jet')
                axes[1, j+1].set_title(f'{filter_type} - 滤波后幅度谱', fontsize=11)
                axes[1, j+1].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{base_name}_{noise_type}_spectrum_filtering.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")

def frequency_domain_filtering(image, filter_types=['lowpass', 'highpass'], 
                              use_gaussian=True, lowpass_cutoff=30, 
                              highpass_cutoff=50, band_cutoff=40, band_width=20):
    """
    对单张图像应用多种频域滤波器
    返回: 滤波结果字典
    """
    results = []
    
    for filter_type in filter_types:
        if use_gaussian:
            if filter_type in ['lowpass', 'highpass']:
                cutoff = lowpass_cutoff if filter_type == 'lowpass' else highpass_cutoff
                mask = create_gaussian_filter(image.shape[:2], filter_type, cutoff)
            else:
                mask = create_gaussian_filter(image.shape[:2], filter_type, band_cutoff, band_width)
        else:
            if filter_type in ['lowpass', 'highpass']:
                cutoff = lowpass_cutoff if filter_type == 'lowpass' else highpass_cutoff
                mask = create_ideal_filter(image.shape[:2], filter_type, cutoff)
            else:
                mask = create_ideal_filter(image.shape[:2], filter_type, band_cutoff, band_width)
        
        filtered_img = apply_frequency_filter(image, mask)
        results.append(filtered_img)
    
    return results

def process_images(misc_noise_dir='misc-noise', result_dir='result2'):
    """
    处理所有图片，应用频域滤波
    """
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    
    # 定义要使用的滤波器
    filter_types = ['lowpass', 'highpass', 'bandreject']
    filter_names = ['高斯低通滤波', '高斯高通滤波', '高斯带阻滤波']
    
    # 获取原图列表
    tiff_files = [f for f in os.listdir(misc_noise_dir) if f.endswith('.tiff')]
    original_files = [f for f in tiff_files if '_gauss' not in f and '_pepper' not in f]
    
    if not original_files:
        print("未找到原图文件")
        return
    
    print(f"找到 {len(original_files)} 个原图文件")
    print("="*50)
    
    # 滤波器参数
    filter_params = {
        'lowpass_cutoff': 30,   # 低通截止频率
        'highpass_cutoff': 50,  # 高通截止频率
        'band_cutoff': 40,      # 带通/带阻中心频率
        'band_width': 20        # 带通/带阻宽度
    }
    
    for orig_file in original_files:
        orig_path = os.path.join(misc_noise_dir, orig_file)
        base_name = orig_file.replace('.tiff', '')
        
        # 读取原图
        original = cv2.imread(orig_path)
        if original is None:
            print(f"无法读取原图: {orig_path}")
            continue
        
        # 读取噪声图
        gauss_file = f"{base_name}_gauss.tiff"
        pepper_file = f"{base_name}_pepper.tiff"
        gauss_path = os.path.join(misc_noise_dir, gauss_file)
        pepper_path = os.path.join(misc_noise_dir, pepper_file)
        
        gauss_noisy = cv2.imread(gauss_path) if os.path.exists(gauss_path) else None
        pepper_noisy = cv2.imread(pepper_path) if os.path.exists(pepper_path) else None
        
        if gauss_noisy is None and pepper_noisy is None:
            print(f"未找到 {base_name} 的噪声图，跳过")
            continue
        
        print(f"\n正在处理: {base_name}")
        
        # 收集所有要处理的图像
        images_to_filter = {'original': original}
        noise_types = []
        noisy_images = []
        
        if gauss_noisy is not None:
            images_to_filter['高斯噪声'] = gauss_noisy
            noise_types.append('高斯噪声')
            noisy_images.append(gauss_noisy)
            
        if pepper_noisy is not None:
            images_to_filter['椒盐噪声'] = pepper_noisy
            noise_types.append('椒盐噪声')
            noisy_images.append(pepper_noisy)
        
        # 创建滤波器
        sample_shape = original.shape[:2]
        masks = []
        for filter_type in filter_types:
            if filter_type in ['lowpass', 'highpass']:
                cutoff = filter_params['lowpass_cutoff'] if filter_type == 'lowpass' else filter_params['highpass_cutoff']
            else:
                cutoff = filter_params['band_cutoff']
            mask = create_gaussian_filter(sample_shape, filter_type, cutoff, filter_params['band_width'])
            masks.append(mask)
        
        
        
        # 对所有图像应用滤波
        filtered_results = {}
        for img_type, img in images_to_filter.items():
            filtered_results[img_type] = frequency_domain_filtering(
                img, filter_types, use_gaussian=True, **filter_params
            )
            print(f"  - 完成 {img_type} 的频域滤波")
        
        # 创建综合对比图
        create_comprehensive_comparison(
            original, noisy_images, filtered_results, 
            noise_types, filter_names, base_name, result_dir
        )
        
        # 创建频谱对比图
        create_spectrum_comparison(
            original, noisy_images, filtered_results,
            noise_types, filter_names, base_name, result_dir
        )
        
        print(f"完成 {base_name} 的处理")
        print("-"*50)
    
    print(f"\n所有处理完成！结果保存在 {result_dir} 文件夹中。")

def main():

    print("\n使用滤波器：")
    print("1. 高斯低通滤波器 - 去除高频噪声（椒盐噪声有效）")
    print("2. 高斯高通滤波器 - 去除低频变化（高斯噪声可能有效）")
    print("3. 高斯带阻滤波器 - 去除特定频率范围的噪声")
    print("="*50)
    
    process_images(misc_noise_dir='misc-noise', result_dir='result2')
    
    print("\n生成的文件包括：")
    print("  - *_frequency_filter_comparison.png (滤波效果综合对比)")
    print("  - *_*_spectrum_filtering.png (滤波前后频谱对比)")

if __name__ == '__main__':
    main()