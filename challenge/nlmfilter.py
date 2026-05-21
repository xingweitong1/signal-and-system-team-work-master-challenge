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

GAUSS_SIGMA = 25
PEPPER_PROB = 0.02
RANDOM_SEED = 42

# 混合噪声两阶段去噪：中值去椒盐 + NLM 去高斯
DEFAULT_H = 19
DEFAULT_H_COLOR = 12
DEFAULT_TEMPLATE_SIZE = 7
DEFAULT_SEARCH_SIZE = 21
MEDIAN_PRE_KERNEL = 3

TITLE_FONTSIZE = 11
SUPTITLE_FONTSIZE = 15


def _nlm_params_for_shape(image_shape, h=DEFAULT_H, h_color=DEFAULT_H_COLOR):
    height, width = image_shape[:2]
    short_side = min(height, width)

    if short_side >= 1200:
        return h + 2, h_color + 1, 9, 25
    if short_side >= 700:
        return h + 1, h_color, 7, 21
    return h, h_color, 7, 21

# 生成混合噪声图像
def add_gaussian_noise(img, sigma=GAUSS_SIGMA):
    """添加高斯噪声"""
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(img, prob=PEPPER_PROB):
    """添加椒盐噪声"""
    noisy_img = img.copy()
    salt_mask = np.random.random(img.shape[:2]) < prob / 2
    pepper_mask = np.random.random(img.shape[:2]) < prob / 2

    if len(img.shape) == 3:
        salt_mask = np.stack([salt_mask] * img.shape[2], axis=2)
        pepper_mask = np.stack([pepper_mask] * img.shape[2], axis=2)

    noisy_img[salt_mask] = 255
    noisy_img[pepper_mask] = 0
    return noisy_img


def add_mixed_noise(img):
    """混合噪声：高斯噪声 + 椒盐噪声"""
    return add_salt_pepper_noise(add_gaussian_noise(img))


def to_uint8(image):
    """NLM 需要 uint8 输入"""
    if image.dtype == np.uint8:
        return image
    if image.dtype == np.uint16:
        return (image / 256).astype(np.uint8)
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def nlm_denoise(image, h=DEFAULT_H, h_color=DEFAULT_H_COLOR,
                template_window_size=DEFAULT_TEMPLATE_SIZE,
                search_window_size=DEFAULT_SEARCH_SIZE,
                median_pre_kernel=MEDIAN_PRE_KERNEL):
    """
    混合噪声去噪：先中值滤波去除椒盐脉冲，再 NLM 抑制高斯噪声
    """
    img = to_uint8(image)
    h, h_color, template_window_size, search_window_size = _nlm_params_for_shape(
        img.shape, h=h, h_color=h_color
    )

    if median_pre_kernel and median_pre_kernel >= 3:
        img = cv2.medianBlur(img, median_pre_kernel)

    if len(img.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(
            img, None, h, h_color, template_window_size, search_window_size
        )
    return cv2.fastNlMeansDenoising(
        img, None, h, template_window_size, search_window_size
    )


def mean_filter(image, kernel_size=3):
    return cv2.blur(image, (kernel_size, kernel_size))


def median_filter(image, kernel_size=3):
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(image, kernel_size)


def show_image(ax, img, title):
    if len(img.shape) == 2:
        ax.imshow(img, cmap='gray')
    else:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=8)
    ax.axis('off')


def create_nlm_comparison_figure(original, nlm_result, median_result, mean_result,
                                 base_name, save_path):
    """混合噪声去噪对比图：原图与各滤波结果"""
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(f'{base_name} - 混合噪声NLM去噪对比', fontsize=SUPTITLE_FONTSIZE, fontweight='bold', y=1.02)

    col_titles = ['原图 ', 'NLM非局部均值', '中值滤波', '均值滤波']


    show_image(axes[0], original, '混合噪声\n' + col_titles[0])
    show_image(axes[1], nlm_result, '混合噪声\n' + col_titles[1])
    show_image(axes[2], median_result, '混合噪声\n' + col_titles[2])
    show_image(axes[3], mean_result, '混合噪声\n' + col_titles[3])

    plt.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.02, wspace=0.08)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_images(misc_noise_dir=None, result_dir=None,
                   h=DEFAULT_H, h_color=DEFAULT_H_COLOR):
    """从原图生成混合噪声，并用 NLM 等方法去噪"""
    if misc_noise_dir is None:
        misc_noise_dir = MISC_NOISE_DIR
    if result_dir is None:
        result_dir = RESULT_DIR

    Path(result_dir).mkdir(parents=True, exist_ok=True)

    if not os.path.isdir(misc_noise_dir):
        print("未找到 misc-noise 文件夹，请先运行 noiseadding.py")
        return

    tiff_files = [f for f in os.listdir(misc_noise_dir) if f.endswith('.tiff')]
    original_files = [f for f in tiff_files if '_gauss' not in f and '_pepper' not in f and '_mixed' not in f]

    if not original_files:
        return

    np.random.seed(RANDOM_SEED)

    for index, orig_file in enumerate(original_files, start=1):
        orig_path = os.path.join(misc_noise_dir, orig_file)
        base_name = orig_file.replace('.tiff', '')

        original = cv2.imread(orig_path)
        if original is None:
            continue

        print(f"[{index}/{len(original_files)}] 正在处理 {orig_file}")

        mixed_noisy = add_mixed_noise(original)
        mixed_path = os.path.join(result_dir, f"{base_name}_mixed.tiff")
        cv2.imwrite(mixed_path, mixed_noisy)

        nlm_result = nlm_denoise(mixed_noisy, h=h, h_color=h_color)
        median_result = median_filter(mixed_noisy)
        mean_result = mean_filter(mixed_noisy)

        create_nlm_comparison_figure(
            original, nlm_result, median_result, mean_result,
            base_name,
            os.path.join(result_dir, f"{base_name}_nlm_comparison.png")
        )


def main():
    process_images()
    print(f"NLM混合噪声去噪完成，结果已保存至 {RESULT_DIR}")


if __name__ == '__main__':
    main()
