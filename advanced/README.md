# 进阶任务：双边滤波（保边去噪）

## 项目功能

### 双边滤波 (bilateralfilter.py)
- 实现双边滤波算法，在去噪的同时更好地保留边缘细节
- 与高斯噪声、椒盐噪声图像上的均值滤波进行对比
- 分析 `sigmaColor`（灰度相似度）与 `sigmaSpace`（空间范围）对去噪与保边的影响
- 计算 PSNR、SSIM 量化指标


## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 准备数据
确保上级目录已存在 `misc-noise` 文件夹（由 `noiseadding.py` 生成）。

### 2. 运行双边滤波脚本

```bash
cd advanced
python bilateralfilter.py
```

结果保存在 `advanced/result/` 文件夹中。

## 文件结构

```
advanced/
├── bilateralfilter.py   # 双边滤波主脚本
├── requirements.txt     # Python 依赖
├── README.md            # 说明文档
└── result/              # 输出结果（运行后生成）
    ├── *_bilateral_comparison.png
    └── *_parameter_analysis.png
```


## 参数选择讨论

- **σ_r 较小**：仅灰度相近的像素参与加权，边缘处权重迅速衰减，保边效果好，但去噪能力有限
- **σ_r 较大**：更多像素参与平均，去噪增强，但边缘会被平滑
- **σ_s 较小**：仅考虑近邻像素，局部性强，细节保留好
- **σ_s 较大**：远距离像素也参与，整体更平滑，易损失纹理

## 技术特点

- 空域非线性滤波，结合高斯空间核与灰度值域核
- 参数扫描可视化，便于理解去噪与保边的权衡
