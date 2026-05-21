# 信号与系统大作业---图像噪声处理：基础部分

## 项目功能

### 1. 噪声添加 (noiseadding.py)
- 添加高斯噪声到图像
- 添加椒盐噪声到图像
- 支持批量处理文件夹中的图像

### 2. 空间域滤波 (meanandmedianfilter.py)
- 均值滤波：适合去除高斯噪声
- 中值滤波：适合去除椒盐噪声
- 生成滤波效果对比图

### 3. 频域分析 (DFTanalysis.py)
- 离散傅里叶变换 (DFT)
- 幅度谱和相位谱计算
- 频谱差异分析
- 频率剖面分析

### 4. 频域滤波 (frequencyfilter.py)
- 理想低通/高通滤波器
- 高斯低通/高通滤波器
- 带通/带阻滤波器
- 频域滤波效果对比分析

### 5. 高级频域滤波 (frequencyfilter2.py)
- 维纳滤波：自适应滤波，特别适合高斯噪声
- 巴特沃斯低通滤波器：平滑截止特性
- PSNR和SSIM质量评估



## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 准备数据
将原始图像文件（.tiff格式）放入 `misc` 文件夹中。

### 2. 添加噪声
运行噪声添加脚本：
```python
python noiseadding.py
```
这将在 `misc-noise` 文件夹中生成添加了高斯噪声和椒盐噪声的图像文件。

### 3. 空间域滤波
运行空间域滤波脚本：
```python
python meanandmedianfilter.py
```
将在 `result` 文件夹中生成滤波效果对比图。

### 4. 频域分析
运行频域分析脚本：
```python
python DFTanalysis.py
```
将在相应文件夹中生成频域分析图表。

### 5. 频域滤波
运行频域滤波脚本：
```python
python frequencyfilter.py
```
或
```python
python frequencyfilter2.py
```
将在 `result1`、`result2`、`result3` 文件夹中生成频域分析结果。



## 文件结构

```
├── DFTanalysis.py          # 频域分析脚本
├── frequencyfilter.py      # 基础频域滤波
├── frequencyfilter2.py     # 高级频域滤波
├── meanandmedianfilter.py  # 空间域滤波
├── noiseadding.py          # 噪声添加脚本
├── requirements.txt        # Python依赖
├── misc/                   # 原始图像文件夹
├── misc-noise/            # 噪声图像文件夹
├── result/                # 空间域滤波结果
├── result1/               # 频域滤波结果1
├── result2/               # 频域滤波结果2
└── result3/               # 频域滤波结果3
```

## 技术特点

- **多类型噪声处理**：支持高斯噪声和椒盐噪声
- **多域滤波**：空间域和频域结合
- **质量评估**：PSNR和SSIM指标计算
- **可视化分析**：丰富的频谱和对比图表
- **批量处理**：支持文件夹批量处理

## 注意事项

- 图像格式：主要支持.tiff格式
- 依赖库：需要OpenCV、NumPy、Matplotlib
- 输出格式：结果主要保存为PNG图像文件
