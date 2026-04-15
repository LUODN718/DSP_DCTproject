# PyTorch: Imagenette + DCT

一个用 PyTorch 训练流程和图像频域处理的小项目，包含：

- 基于 `imagenette2-160` 的简单 CNN 分类训练
- 频域可视化工具（FFT 幅度谱、8x8 Block DCT、低频重建）

## 项目结构

- `train_classifier.py`：分类模型训练脚本（支持可选 DCT 预处理）
- `visualize_spectrum.py`：频域可视化脚本
- `test.py`：加载 `tiny_imagenette.pt` 做单张图片推理（中文类别名输出）
- `requirements.txt`：Python 依赖
- `imagenette2-160/`：训练与验证数据集

## 环境准备

在项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 数据集

训练脚本默认读取项目根目录下的：

`imagenette2-160/train`
`imagenette2-160/val`

如果目录名不变，无需额外参数。

## 训练模型

### 1) 常规训练（RGB 输入）

```bash
source .venv/bin/activate
python train_classifier.py
```

可指定训练轮数：

```bash
python train_classifier.py --epochs 1
```

### 2) 使用 Block DCT 预处理训练

`train_classifier.py` 中，DCT 作为独立预处理模块，CNN 结构保持不变。可选两种模式：

- `coeff`：直接使用 DCT 系数域输入
- `recon_lowfreq`：仅保留低频后 IDCT 回空间域输入

```bash
python train_classifier.py --use_dct --dct_mode coeff --dct_block 8 --epochs 10
python train_classifier.py --use_dct --dct_mode recon_lowfreq --dct_block 8 --dct_keep 4 --epochs 10
```

训练后会在根目录生成权重文件：

- `tiny_imagenette.pt`（常规训练）
- `tiny_imagenette_dct8.pt`（DCT 训练）

## 模型推理

`test.py` 会加载 `tiny_imagenette.pt` 并输出中文类别名。使用前先把脚本中的图片路径改成你自己的本地图片：

```python
img = Image.open("本地图片路径").convert("RGB")
```

运行命令：

```bash
source .venv/bin/activate
python test.py
```

示例输出：

```text
预测类别: 英国史宾格犬 (n02102040)
```

## 频域可视化

示例命令：

```bash
python visualize_spectrum.py \
  --image imagenette2-160/val/n02102040/ILSVRC2012_val_00008162.JPEG \
  --out outputs/spectrum_visualization.png
```

输出图会展示：

- 原图灰度图
- 低频保留后重建图（处理后）

## 说明

- 设备自动选择：优先 `mps`（Apple Silicon），其次 `cuda`，否则 `cpu`
- 这是学习性质的基础工程，模型结构简洁，便于理解训练和频域处理流程
