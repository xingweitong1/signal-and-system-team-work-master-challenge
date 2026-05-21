# 挑战任务：非局部均值(NLM)混合噪声去噪

## 项目功能

### NLM 去噪 (nlmfilter.py)
- 从 `misc-noise` 原图生成混合噪声（高斯 + 椒盐）并保存
- 实现非局部均值(NLM)去噪算法
- 对混合噪声图去噪，与均值滤波、中值滤波对比
- 计算 PSNR、SSIM 量化评估
- 生成去噪对比图与指标柱状图

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 准备数据
确保上级目录已存在 `misc-noise` 文件夹（由 `noiseadding.py` 生成）。

### 2. 运行 NLM 去噪脚本

```bash
cd challenge
python nlmfilter.py
```

脚本会从 `misc-noise` 读取原图、生成混合噪声并去噪，结果保存在 `challenge/result/` 文件夹中。

## 文件结构

```
challenge/
├── nlmfilter.py         # NLM 去噪主脚本
├── requirements.txt     # Python 依赖
├── latexcode.txt        # LaTeX 公式与报告代码
├── README.md            # 说明文档
└── result/              # 输出结果（运行后生成）
    ├── *_mixed.tiff
    └── *_nlm_comparison.png
```

## 混合噪声模型

在原图上依次叠加高斯噪声（σ=25）与椒盐噪声（2%），生成 `*_mixed.tiff` 后进行 NLM 去噪。



## 技术特点

- 针对混合噪声设计，NLM 利用图像自相似性
- 与均值、中值滤波横向对比
