"""
test_pi.py
================
Phase 2 PSNR/SSIM validation on Pi: load GPU-trained checkpoint, reproduce
paper numbers using SFMformer's exact patch-wise inference and BasicSR's
exact PSNR/SSIM formulas.

Two key behaviours that must mirror the original training/testing pipeline:

  1. Patch-wise inference with overlap, exactly as `SFMformerModel.test()` does.
  2. PSNR / SSIM identical to BasicSR's `calculate_psnr` / `calculate_ssim`
     with `test_y_channel=True` (the standard SR community convention).

Usage
-----
    python test_pi.py --scale 2                    # Set5 x2 default
    python test_pi.py --scale 2 --benchmark Set14
    python test_pi.py --scale 4 --save             # also save SR images
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))


# =============================================================================
# Model config -- mirrors options/test/101_SFMformer_SRx2_scratch.yml exactly.
# =============================================================================
SFMFORMER_LIGHT_CONFIG = dict(
    in_chans=3,
    img_size=64,
    embed_dim=52,
    depths=[2, 4, 6, 6, 6],
    num_heads=4,
    num_topk=[
        1024, 1024,
        256, 256, 256, 256,
        128, 128, 128, 128, 128, 128,
        64, 64, 64, 64, 64, 64,
        32, 32, 32, 32, 32, 32,
    ],
    window_size=32,
    convffn_kernel_size=7,
    img_range=1.0,
    mlp_ratio=1,
    upsampler='pixelshuffledirect',
    resi_connection='1conv',
    use_checkpoint=False,
    use_ups=True,
)

CHECKPOINT_MAP = {
    2: '101_SFMformer_SRx2_scratch.pth',
    3: '102_SFMformer_SRx3_finetune.pth',
    4: '103_SFMformer_SRx4_finetune.pth',
}


# =============================================================================
# Image <-> tensor helpers
# =============================================================================
def img2tensor(img_uint8_rgb: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img_uint8_rgb).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0)


def tensor2img(t: torch.Tensor) -> np.ndarray:
    arr = t.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (arr * 255.0).round().astype(np.uint8)


# =============================================================================
# BasicSR-exact PSNR
# =============================================================================
def to_y_channel(img_uint8_rgb: np.ndarray) -> np.ndarray:
    img = img_uint8_rgb.astype(np.float64) / 255.0
    return np.dot(img, [65.481, 128.553, 24.966]) + 16.0


def calculate_psnr(img: np.ndarray, img2: np.ndarray,
                   crop_border: int, test_y_channel: bool = False) -> float:
    assert img.shape == img2.shape, f'shape mismatch: {img.shape} vs {img2.shape}'
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    if crop_border > 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    if test_y_channel:
        img = to_y_channel(img.astype(np.uint8))
        img2 = to_y_channel(img2.astype(np.uint8))
    mse = np.mean((img - img2) ** 2)
    if mse < 1e-10:
        return float('inf')
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


# =============================================================================
# BasicSR-exact SSIM (Wang 2004 with 11x11 Gaussian sigma=1.5)
# =============================================================================
def _ssim_single_channel(img: np.ndarray, img2: np.ndarray) -> float:
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img: np.ndarray, img2: np.ndarray,
                   crop_border: int, test_y_channel: bool = False) -> float:
    assert img.shape == img2.shape, f'shape mismatch: {img.shape} vs {img2.shape}'
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    if crop_border > 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        y1 = to_y_channel(img.astype(np.uint8))
        y2 = to_y_channel(img2.astype(np.uint8))
        return _ssim_single_channel(y1, y2)
    else:
        ssims = [_ssim_single_channel(img[..., c], img2[..., c])
                 for c in range(img.shape[2])]
        return float(np.mean(ssims))


# =============================================================================
# Patch-wise inference, ported from SFMformerModel.test()
# =============================================================================
@torch.no_grad()
def patchwise_test(model: torch.nn.Module, lq: torch.Tensor,
                   scale: int) -> torch.Tensor:
    _, C, h, w = lq.size()
    split_token_h = h // 256 + 1
    split_token_w = w // 256 + 1

    mod_pad_h = (split_token_h - h % split_token_h) % split_token_h
    mod_pad_w = (split_token_w - w % split_token_w) % split_token_w
    img = F.pad(lq, (0, mod_pad_w, 0, mod_pad_h), mode='reflect')
    _, _, H, W = img.size()

    split_h = H // split_token_h
    split_w = W // split_token_w
    shave_h = split_h // 10
    shave_w = split_w // 10
    ral = H // split_h
    row = W // split_w

    slices = []
    for i in range(ral):
        for j in range(row):
            if i == 0 and i == ral - 1:
                top = slice(i * split_h, (i + 1) * split_h)
            elif i == 0:
                top = slice(i * split_h, (i + 1) * split_h + shave_h)
            elif i == ral - 1:
                top = slice(i * split_h - shave_h, (i + 1) * split_h)
            else:
                top = slice(i * split_h - shave_h, (i + 1) * split_h + shave_h)

            if j == 0 and j == row - 1:
                left = slice(j * split_w, (j + 1) * split_w)
            elif j == 0:
                left = slice(j * split_w, (j + 1) * split_w + shave_w)
            elif j == row - 1:
                left = slice(j * split_w - shave_w, (j + 1) * split_w)
            else:
                left = slice(j * split_w - shave_w, (j + 1) * split_w + shave_w)
            slices.append((top, left))

    outputs = [model(img[..., t, l]) for (t, l) in slices]

    out = torch.zeros(1, C, H * scale, W * scale, dtype=img.dtype, device=img.device)
    for i in range(ral):
        for j in range(row):
            top  = slice(i * split_h * scale, (i + 1) * split_h * scale)
            left = slice(j * split_w * scale, (j + 1) * split_w * scale)
            _top  = slice(0, split_h * scale)             if i == 0 else slice(shave_h * scale, (shave_h + split_h) * scale)
            _left = slice(0, split_w * scale)             if j == 0 else slice(shave_w * scale, (shave_w + split_w) * scale)
            out[..., top, left] = outputs[i * row + j][..., _top, _left]

    out = out[:, :, : H * scale - mod_pad_h * scale, : W * scale - mod_pad_w * scale]
    return out


# =============================================================================
# Checkpoint loading
# =============================================================================
def load_state_dict_flexibly(model: torch.nn.Module, ckpt_path: str) -> None:
    raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(raw, dict):
        for key in ('params_ema', 'params', 'state_dict', 'model'):
            if key in raw:
                sd = raw[key]
                print(f"  Found state_dict under key: '{key}'")
                break
        else:
            sd = raw
            print('  Loading raw state_dict (no params wrapper)')
    else:
        sd = raw

    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    try:
        model.load_state_dict(sd, strict=True)
        print('  Strict load OK -- all keys matched perfectly')
    except RuntimeError as e:
        print(f'  Strict load failed:\n  {str(e)[:300]}...\n  Falling back to strict=False')
        model.load_state_dict(sd, strict=False)


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scale', type=int, default=2, choices=[2, 3, 4])
    ap.add_argument('--benchmark', type=str, default='Set5',
                    choices=['Set5', 'Set14', 'B100', 'Urban100', 'Manga109'])
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--save', action='store_true')
    ap.add_argument('--root', type=str, default=str(Path(__file__).parent))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    root = Path(args.root)
    ckpt_path = root / 'experiments' / 'pretrained_models' / CHECKPOINT_MAP[args.scale]
    hr_dir = root / 'datasets' / 'TestDataSR' / 'HR' / args.benchmark / f'x{args.scale}'
    lr_dir = root / 'datasets' / 'TestDataSR' / 'LR' / 'LRBI' / args.benchmark / f'x{args.scale}'
    save_dir = root / 'results' / f'{args.benchmark}_x{args.scale}'
    if args.save:
        save_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 84)
    print(f'Phase 2 Validation -- {args.benchmark} x{args.scale} (patch-wise, BasicSR-equivalent PSNR/SSIM)')
    print('=' * 84)
    print(f'Checkpoint : {ckpt_path}')
    print(f'HR dir     : {hr_dir}')
    print(f'LR dir     : {lr_dir}')
    print(f'CPU threads: {args.threads}\n')

    for p, name in [(ckpt_path, 'checkpoint'), (hr_dir, 'HR dir'), (lr_dir, 'LR dir')]:
        if not p.exists():
            print(f'[ERROR] {name} not found: {p}')
            return

    print('[1/3] Building model...')
    from basicsr.archs.sfmformer_arch import SFMformer
    model = SFMformer(upscale=args.scale, **SFMFORMER_LIGHT_CONFIG)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters: {n_params/1e3:.1f}K ({n_params/1e6:.2f}M)')

    print('[2/3] Loading checkpoint...')
    load_state_dict_flexibly(model, str(ckpt_path))
    model.eval()

    print(f'\n[3/3] Running patch-wise inference on {args.benchmark}...\n')
    print(f'{"image":<14}{"LR size":<14}{"SR size":<14}{"time":>8}    '
          f'{"PSNR-Y":>8}   {"SSIM-Y":>8}   {"PSNR-RGB":>8}   {"SSIM-RGB":>8}')
    print('-' * 96)

    lr_files = sorted([p for p in lr_dir.iterdir()
                       if p.suffix.lower() in {'.png', '.bmp', '.jpg', '.jpeg'}])
    if not lr_files:
        print('[ERROR] No images in LR dir')
        return

    psnr_y_l, ssim_y_l, psnr_rgb_l, ssim_rgb_l, time_l = [], [], [], [], []

    for lr_file in lr_files:
        hr_file = hr_dir / lr_file.name
        if not hr_file.exists():
            print(f'  [WARN] {lr_file.name}: no matching HR, skipping')
            continue

        lr_img = np.array(Image.open(lr_file).convert('RGB'))
        hr_img = np.array(Image.open(hr_file).convert('RGB'))
        lr_t = img2tensor(lr_img)

        t0 = time.perf_counter()
        sr_t = patchwise_test(model, lr_t, scale=args.scale)
        dt = time.perf_counter() - t0

        sr_img = tensor2img(sr_t)
        h = min(sr_img.shape[0], hr_img.shape[0])
        w = min(sr_img.shape[1], hr_img.shape[1])
        sr_img = sr_img[:h, :w]
        hr_img = hr_img[:h, :w]

        psnr_y   = calculate_psnr(sr_img, hr_img, crop_border=args.scale, test_y_channel=True)
        ssim_y   = calculate_ssim(sr_img, hr_img, crop_border=args.scale, test_y_channel=True)
        psnr_rgb = calculate_psnr(sr_img, hr_img, crop_border=args.scale, test_y_channel=False)
        ssim_rgb = calculate_ssim(sr_img, hr_img, crop_border=args.scale, test_y_channel=False)

        psnr_y_l.append(psnr_y); ssim_y_l.append(ssim_y)
        psnr_rgb_l.append(psnr_rgb); ssim_rgb_l.append(ssim_rgb)
        time_l.append(dt)

        print(f'{lr_file.stem:<14}{str(tuple(lr_t.shape[2:])):<14}{str(sr_img.shape[:2]):<14}'
              f'{dt:>6.2f}s    {psnr_y:>7.4f}   {ssim_y:>7.4f}   {psnr_rgb:>7.4f}   {ssim_rgb:>7.4f}')

        if args.save:
            Image.fromarray(sr_img).save(save_dir / f'{lr_file.stem}_SR.png')

    print('-' * 96)
    print(f'{"AVERAGE":<14}{"":<14}{"":<14}{np.mean(time_l):>6.2f}s    '
          f'{np.mean(psnr_y_l):>7.4f}   {np.mean(ssim_y_l):>7.4f}   '
          f'{np.mean(psnr_rgb_l):>7.4f}   {np.mean(ssim_rgb_l):>7.4f}')
    print()
    print(f'Compare AVERAGE PSNR-Y / SSIM-Y against your paper\'s Table.')
    print('Match within 0.01 dB / 0.0005 SSIM confirms CPU deployment is correct.')
    if args.save:
        print(f'\nSR images saved to: {save_dir}')


if __name__ == '__main__':
    main()