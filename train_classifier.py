#!/usr/bin/env python3
"""
在你本机的 Imagenette（10 类小图）上训练一个简单 CNN，边写边对照 PyTorch 核心概念。
依赖: pip install -r requirements.txt（建议在项目目录下用 python -m venv .venv 建虚拟环境）
运行示例（请只复制命令本身，不要带行尾说明；若 # 不是英文半角，shell 不会当注释，会传给 Python 报错）:

  source .venv/bin/activate
  python train_classifier.py
  python train_classifier.py --epochs 1
"""

from __future__ import annotations
import argparse
import math
import pathlib
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

# -----------------------------------------------------------------------------
# 选择设备部分（在mac上是mps，有cuda用cuda）
# PyTorch 里张量与模型要放在同一设备上才能运算。Apple Silicon Mac 上常用 MPS；
# 没有则退回 CPU。cuda 在 Mac 上一般不可用。
# -----------------------------------------------------------------------------
def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# Imagenette 10 类的可读名字（文件夹名是 WordNet id）
CLASS_NAMES = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}


# -----------------------------------------------------------------------------
# 2) 数据与 DataLoader
# - Dataset: 描述「第 i 个样本是什么」；ImageFolder 按子文件夹名作为类别。
# - transforms: 把 PIL 图变成张量并做归一化；训练集常加随机增强。
# - DataLoader: 批采样、打乱、多进程 num_workers（Mac 上 0 往往更稳）。
# -----------------------------------------------------------------------------
class BlockDCTTransform:
    """
    对每张图做“8x8 block DCT”，输出与输入同形状：[C, H, W]。
    - 输入：PIL image
    - 输出：DCT 系数（每通道做标准化）
    """

    def __init__(self, block_size: int = 8, standardize: bool = True) -> None:
        self.block_size = int(block_size)
        self.standardize = standardize
        self._build_dct_matrices()

    def _build_dct_matrices(self) -> None:
        N = self.block_size
        C = torch.zeros(N, N, dtype=torch.float32)
        for k in range(N):
            alpha = (1.0 / N) ** 0.5 if k == 0 else (2.0 / N) ** 0.5
            for n in range(N):
                # 这里是一次性预计算常量，用 math.cos 更直接
                angle = math.pi * (n + 0.5) * k / N
                C[k, n] = alpha * math.cos(angle)
        self.C = C
        self.C_t = C.t()

    def __call__(self, img) -> torch.Tensor:
        x = TF.to_tensor(img).to(torch.float32)  # [ch, H, W]
        ch, H, W = x.shape
        N = self.block_size
        if H % N != 0 or W % N != 0:
            raise ValueError(f"DCT 需要 H/W 能被 block_size 整除，但得到 H={H}, W={W}, block_size={N}")

        Hb, Wb = H // N, W // N
        # [ch, Hb, Wb, N, N]
        blocks = x.view(ch, Hb, N, Wb, N).permute(0, 1, 3, 2, 4).contiguous()
        # 展平成批：B=ch*Hb*Wb
        blocks2 = blocks.view(ch * Hb * Wb, N, N)

        # y = C @ x @ C^T （在每个 8x8 块上做）
        tmp = torch.matmul(self.C, blocks2)  # [B, N, N]
        dct2 = torch.matmul(tmp, self.C_t)   # [B, N, N]

        # 还原形状：[ch, Hb, Wb, N, N] -> [ch, H, W]
        out = dct2.view(ch, Hb, Wb, N, N).permute(0, 1, 3, 2, 4).contiguous().view(ch, H, W)

        if self.standardize:
            mean = out.mean(dim=(1, 2), keepdim=True)
            std = out.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            out = (out - mean) / std
        return out


def make_dataloaders(
    data_root: pathlib.Path,
    batch_size: int,
    num_workers: int,
    use_dct: bool,
    dct_block: int,
) -> tuple[DataLoader, DataLoader, list[str]]:
    img_size = 160  # imagenette2-160；保证能被 dct_block 整除
    if use_dct and img_size % dct_block != 0:
        raise ValueError(f"img_size={img_size} 需要能被 dct_block={dct_block} 整除")

    if use_dct:
        train_tfms = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                BlockDCTTransform(block_size=dct_block, standardize=True),
            ]
        )
        val_tfms = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                BlockDCTTransform(block_size=dct_block, standardize=True),
            ]
        )
    else:
        train_tfms = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        val_tfms = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    train_set = datasets.ImageFolder(root=data_root / "train", transform=train_tfms)
    val_set = datasets.ImageFolder(root=data_root / "val", transform=val_tfms)
    # class_to_idx: 文件夹名 -> 0..C-1，训练时要和标签一致
    idx_to_folder = sorted(train_set.class_to_idx, key=lambda k: train_set.class_to_idx[k])

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader, idx_to_folder


# -----------------------------------------------------------------------------
# 3) 模型（nn.Module）
# - 在 __init__ 里定义层；在 forward 里写数据怎样流过这些层。
# - 最后 Linear 输出长度为 num_classes；多分类与 CrossEntropyLoss 配合时不再套 Softmax。
# -----------------------------------------------------------------------------
class TinyCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# -----------------------------------------------------------------------------
# 4) 训练一步在做什么（核心循环）
# - loss = criterion(outputs, targets)：outputs 是 logits，targets 是类别索引 long。
# - loss.backward()：自动求导，把梯度写到每个参数的 .grad。
# - optimizer.step()：按梯度更新参数；之后 optimizer.zero_grad() 清空梯度。
# - model.train() / model.eval()：影响 Dropout/BatchNorm 等层的行为（本模型无 BN）。
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        loss_sum += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.numel()
    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def print_sample_predictions(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    idx_to_folder: list[str],
    n: int = 8,
) -> None:
    """
    展示「分类模型在干什么」：对验证集里一小批图，打印每张图的真实类别 vs 模型预测的类别。
    训练过程中打印的 loss/acc 是整集统计；这里才是逐样本的「分类结果」。
    """
    model.eval()
    images, targets = next(iter(val_loader))
    images = images.to(device)
    logits = model(images)
    preds = logits.argmax(dim=1)
    print(f"\n--- 验证集抽样：前 {min(n, images.size(0))} 张图的分类结果 ---")
    print("每行：是否猜对 | 真实类别 -> 模型预测类别")
    for i in range(min(n, images.size(0))):
        t, p = targets[i].item(), preds[i].item()
        folder_t, folder_p = idx_to_folder[t], idx_to_folder[p]
        name_t = CLASS_NAMES.get(folder_t, folder_t)
        name_p = CLASS_NAMES.get(folder_p, folder_p)
        ok = "对" if t == p else "错"
        print(f"  [{ok}]  真实: {name_t}  ->  预测: {name_p}")


def train(epochs: int, use_dct: bool, dct_block: int) -> None:
    data_root = pathlib.Path(__file__).resolve().parent / "imagenette2-160"
    if not (data_root / "train").is_dir():
        raise SystemExit(f"找不到训练集目录: {data_root / 'train'}")

    device = pick_device()
    print(f"使用设备: {device}")

    batch_size = 64
    if use_dct:
        # DCT 预处理会增加 CPU 计算量，batch 稍小更稳
        batch_size = 32
    num_workers = 0
    lr = 1e-3

    train_loader, val_loader, idx_to_folder = make_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        use_dct=use_dct,
        dct_block=dct_block,
    )
    num_classes = len(idx_to_folder)
    print(f"类别数: {num_classes}（示例: {idx_to_folder[0]} -> {CLASS_NAMES.get(idx_to_folder[0], '?')}）")

    model = TinyCNN(num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        running_loss = 0.0
        n_samples = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            n_samples += images.size(0)

        train_loss = running_loss / max(n_samples, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)
        dt = time.perf_counter() - t0
        print(
            f"epoch {epoch}/{epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc*100:.1f}%  ({dt:.1f}s)"
        )

    out_path = pathlib.Path(__file__).resolve().parent / (
        f"tiny_imagenette_dct{dct_block}.pt" if use_dct else "tiny_imagenette.pt"
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_to_idx": train_loader.dataset.class_to_idx,
            "idx_to_folder": idx_to_folder,
        },
        out_path,
    )
    print(f"已保存权重到: {out_path}")

    print_sample_predictions(model, val_loader, device, idx_to_folder, n=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Imagenette 简单 CNN 训练示例")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--use_dct", action="store_true", help="在输入前做 block DCT 预处理")
    parser.add_argument("--dct_block", type=int, default=8, help="block DCT 的块大小，默认 8x8")
    args = parser.parse_args()
    train(epochs=args.epochs, use_dct=args.use_dct, dct_block=args.dct_block)
