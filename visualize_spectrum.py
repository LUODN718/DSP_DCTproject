#!/usr/bin/env python3
"""
DSP 可视化模块：RGB、灰度图、DCT 系数域、低频保留重建图。

示例：
  python visualize_spectrum.py --image imagenette2-160/val/n01440764/ILSVRC2012_val_00000293.JPEG
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def to_rgb_array(image_path: pathlib.Path, size: int) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32)
    return arr


def to_gray_array(image_path: pathlib.Path, size: int) -> np.ndarray:
    img = Image.open(image_path).convert("L").resize((size, size))
    arr = np.asarray(img, dtype=np.float32)
    return arr


def fft_magnitude(gray: np.ndarray) -> np.ndarray:
    f = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(f_shift))
    return mag


def dct_matrix(n: int) -> np.ndarray:
    C = np.zeros((n, n), dtype=np.float32)
    for k in range(n):
        alpha = np.sqrt(1.0 / n) if k == 0 else np.sqrt(2.0 / n)
        for i in range(n):
            C[k, i] = alpha * np.cos(np.pi * (i + 0.5) * k / n)
    return C


def block_dct(gray: np.ndarray, block_size: int = 8) -> np.ndarray:
    h, w = gray.shape
    if h % block_size != 0 or w % block_size != 0:
        raise ValueError("height/width must be divisible by block_size")

    C = dct_matrix(block_size)
    Ct = C.T
    out = np.zeros_like(gray, dtype=np.float32)
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            blk = gray[y : y + block_size, x : x + block_size]
            out[y : y + block_size, x : x + block_size] = C @ blk @ Ct
    return out


def block_idct(dct_coeff: np.ndarray, block_size: int = 8) -> np.ndarray:
    h, w = dct_coeff.shape
    C = dct_matrix(block_size)
    Ct = C.T
    out = np.zeros_like(dct_coeff, dtype=np.float32)
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            blk = dct_coeff[y : y + block_size, x : x + block_size]
            out[y : y + block_size, x : x + block_size] = Ct @ blk @ C
    return out


def keep_low_freq(dct_coeff: np.ndarray, block_size: int, keep: int) -> np.ndarray:
    h, w = dct_coeff.shape
    out = np.zeros_like(dct_coeff, dtype=np.float32)
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            blk = dct_coeff[y : y + block_size, x : x + block_size]
            low = np.zeros_like(blk)
            low[:keep, :keep] = blk[:keep, :keep]
            out[y : y + block_size, x : x + block_size] = low
    return out


def save_visualization(
    image_path: pathlib.Path,
    output_path: pathlib.Path,
    size: int,
    block_size: int,
    low_keep: int,
) -> None:
    rgb = to_rgb_array(image_path, size=size)
    gray = to_gray_array(image_path, size=size)
    dct_coeff = block_dct(gray, block_size=block_size)
    dct_vis = np.log1p(np.abs(dct_coeff))
    low_dct = keep_low_freq(dct_coeff, block_size=block_size, keep=low_keep)
    recon = block_idct(low_dct, block_size=block_size)
    recon = np.clip(recon, 0, 255)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].imshow(rgb.astype(np.uint8))
    axes[0, 0].set_title("RGB")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gray, cmap="gray")
    axes[0, 1].set_title("Gray")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(dct_vis, cmap="inferno")
    axes[1, 0].set_title(f"DCT Coeff Domain ({block_size}x{block_size}, log)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(recon, cmap="gray")
    axes[1, 1].set_title(f"Low-Freq Recon (keep {low_keep}x{low_keep})", pad=10)
    axes[1, 1].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="DSP spectrum visualization for image")
    parser.add_argument("--image", type=str, required=True, help="input image path")
    parser.add_argument("--size", type=int, default=160, help="resize to size x size")
    parser.add_argument("--block_size", type=int, default=8, help="block size for DCT")
    parser.add_argument("--low_keep", type=int, default=4, help="keep top-left kxk low frequencies")
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/spectrum_visualization.png",
        help="output figure path",
    )
    args = parser.parse_args()

    image_path = pathlib.Path(args.image)
    out_path = pathlib.Path(args.out)
    save_visualization(
        image_path=image_path,
        output_path=out_path,
        size=args.size,
        block_size=args.block_size,
        low_keep=args.low_keep,
    )
    print(f"Saved visualization to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
