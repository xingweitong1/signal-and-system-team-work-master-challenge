import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = Path(__file__).parent
MISC_NOISE_DIR = SCRIPT_DIR.parent / 'misc-noise'
RESULT_DIR = SCRIPT_DIR / 'result'

DEFAULT_D = 9
DEFAULT_SIGMA_COLOR = 75
DEFAULT_SIGMA_SPACE = 75

TITLE_FONTSIZE = 11
SUPTITLE_FONTSIZE = 15


def bilateral_filter(image, d=DEFAULT_D, sigma_color=DEFAULT_SIGMA_COLOR, sigma_space=DEFAULT_SIGMA_SPACE):
    """双边滤波 - 兼顾去噪与保边"""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def mean_filter(image, kernel_size=3):
    """均值滤波（对比基准）"""
    return cv2.blur(image, (kernel_size, kernel_size))


def show_image(ax, img, title):
    """在子图中显示图像"""
    if len(img.shape) == 2:
        ax.imshow(img, cmap='gray')
    else:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=8)
    ax.axis('off')


def compute_psnr(original, filtered):
    """计算PSNR（用于参数分析柱状图）"""
    mse = np.mean((original.astype(np.float32) - filtered.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def compute_ssim(original, filtered):
    """计算SSIM（用于参数分析柱状图）"""
    if len(original.shape) == 3:
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        orig_gray = original

    if len(filtered.shape) == 3:
        filt_gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    else:
        filt_gray = filtered

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    orig_gray = orig_gray.astype(np.float64)
    filt_gray = filt_gray.astype(np.float64)

    mu1 = cv2.GaussianBlur(orig_gray, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(filt_gray, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(orig_gray ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(filt_gray ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(orig_gray * filt_gray, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(ssim_map)


def create_comparison_figure(original, gauss_bilateral, gauss_mean,
                             pepper_bilateral, pepper_mean,
                             base_name, save_path,
                             has_gauss=True, has_pepper=True):
    """对比图：原图与双边/均值滤波结果（不含指标数值）"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{base_name} - 双边滤波效果对比', fontsize=SUPTITLE_FONTSIZE, fontweight='bold', y=0.98)

    col_titles = ['原图 ', '双边滤波', '均值滤波(对比)']


    if has_gauss:
        show_image(axes[0, 0], original, '高斯噪声\n' + col_titles[0])
        show_image(axes[0, 1], gauss_bilateral, '高斯噪声\n双边滤波')
        show_image(axes[0, 2], gauss_mean, '高斯噪声\n均值滤波')
    else:
        for col in range(3):
            axes[0, col].axis('off')

    if has_pepper:
        show_image(axes[1, 0], original, '椒盐噪声\n' + col_titles[0])
        show_image(axes[1, 1], pepper_bilateral, '椒盐噪声\n双边滤波')
        show_image(axes[1, 2], pepper_mean, '椒盐噪声\n均值滤波')
    else:
        for col in range(3):
            axes[1, col].axis('off')

    plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02, wspace=0.08, hspace=0.28)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_parameter_analysis_figure(original, gauss_noisy, base_name, save_dir,
                                     d=DEFAULT_D, sigma_color=DEFAULT_SIGMA_COLOR,
                                     sigma_space=DEFAULT_SIGMA_SPACE):
    """参数分析图：不同 sigma 下的滤波结果与量化柱状图"""
    sigma_color_values = [25, sigma_color, 150]
    sigma_space_values = [25, sigma_space, 150]

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(f'{base_name} - 双边滤波参数对去噪与保边的影响（高斯噪声）',
                 fontsize=SUPTITLE_FONTSIZE, fontweight='bold', y=0.98)

    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.9], hspace=0.32, wspace=0.08)

    fig.text(0.08, 0.72, 'σ_r 变化\n(σ_s固定)', fontsize=TITLE_FONTSIZE, va='center')
    for i, sc in enumerate(sigma_color_values):
        ax = fig.add_subplot(gs[0, i])
        filtered = bilateral_filter(gauss_noisy, d=d, sigma_color=sc, sigma_space=sigma_space)
        show_image(ax, filtered, f'σ_r={sc}')

    fig.text(0.08, 0.42, 'σ_s 变化\n(σ_r固定)', fontsize=TITLE_FONTSIZE, va='center')
    for i, ss in enumerate(sigma_space_values):
        ax = fig.add_subplot(gs[1, i])
        filtered = bilateral_filter(gauss_noisy, d=d, sigma_color=sigma_color, sigma_space=ss)
        show_image(ax, filtered, f'σ_s={ss}')

    ax_bar = fig.add_subplot(gs[2, :])
    labels = []
    psnr_vals = []
    ssim_vals = []

    for sc in sigma_color_values:
        filtered = bilateral_filter(gauss_noisy, d=d, sigma_color=sc, sigma_space=sigma_space)
        labels.append(f'σ_r={sc}')
        psnr_vals.append(compute_psnr(original, filtered))
        ssim_vals.append(compute_ssim(original, filtered))

    for ss in sigma_space_values:
        filtered = bilateral_filter(gauss_noisy, d=d, sigma_color=sigma_color, sigma_space=ss)
        labels.append(f'σ_s={ss}')
        psnr_vals.append(compute_psnr(original, filtered))
        ssim_vals.append(compute_ssim(original, filtered))

    x = np.arange(len(labels))
    width = 0.35
    ax_bar.bar(x - width / 2, psnr_vals, width, label='PSNR (dB)', color='steelblue', alpha=0.8)
    ax_bar.bar(x + width / 2, [s * 100 for s in ssim_vals], width, label='SSIM×100', color='coral', alpha=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels)
    ax_bar.set_title('不同参数下的量化指标（PSNR↑去噪好，SSIM↑保边好）', fontsize=TITLE_FONTSIZE)
    ax_bar.legend(fontsize=10)
    ax_bar.grid(True, alpha=0.3)

    save_path = os.path.join(save_dir, f'{base_name}_parameter_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_images(misc_noise_dir=None, result_dir=None,
                   d=DEFAULT_D, sigma_color=DEFAULT_SIGMA_COLOR, sigma_space=DEFAULT_SIGMA_SPACE):
    """读取 misc-noise 中已生成的高斯/椒盐噪声图进行滤波"""
    if misc_noise_dir is None:
        misc_noise_dir = MISC_NOISE_DIR
    if result_dir is None:
        result_dir = RESULT_DIR

    Path(result_dir).mkdir(parents=True, exist_ok=True)

    if not os.path.isdir(misc_noise_dir):
        print("未找到 misc-noise 文件夹，请先运行 noiseadding.py")
        return

    tiff_files = [f for f in os.listdir(misc_noise_dir) if f.endswith('.tiff')]
    original_files = [f for f in tiff_files if '_gauss' not in f and '_pepper' not in f]

    if not original_files:
        return

    for orig_file in original_files:
        orig_path = os.path.join(misc_noise_dir, orig_file)
        base_name = orig_file.replace('.tiff', '')

        gauss_path = os.path.join(misc_noise_dir, f"{base_name}_gauss.tiff")
        pepper_path = os.path.join(misc_noise_dir, f"{base_name}_pepper.tiff")

        original = cv2.imread(orig_path)
        if original is None:
            continue

        gauss_noisy = cv2.imread(gauss_path) if os.path.exists(gauss_path) else None
        pepper_noisy = cv2.imread(pepper_path) if os.path.exists(pepper_path) else None

        if gauss_noisy is None and pepper_noisy is None:
            continue

        gauss_bilateral = bilateral_filter(gauss_noisy, d, sigma_color, sigma_space) if gauss_noisy is not None else None
        gauss_mean = mean_filter(gauss_noisy) if gauss_noisy is not None else None
        pepper_bilateral = bilateral_filter(pepper_noisy, d, sigma_color, sigma_space) if pepper_noisy is not None else None
        pepper_mean = mean_filter(pepper_noisy) if pepper_noisy is not None else None

        if gauss_noisy is not None or pepper_noisy is not None:
            create_comparison_figure(
                original, gauss_bilateral, gauss_mean,
                pepper_bilateral, pepper_mean,
                base_name,
                os.path.join(result_dir, f"{base_name}_bilateral_comparison.png"),
                has_gauss=gauss_noisy is not None,
                has_pepper=pepper_noisy is not None
            )

        if gauss_noisy is not None:
            create_parameter_analysis_figure(
                original, gauss_noisy, base_name, result_dir,
                d=d, sigma_color=sigma_color, sigma_space=sigma_space
            )


def main():
    process_images()
    print(f"双边滤波处理完成，结果已保存至 {RESULT_DIR}")


if __name__ == '__main__':
    main()
