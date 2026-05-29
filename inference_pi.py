"""
inference_pi.py
================
Single-image super-resolution inference on Raspberry Pi.

Unlike `test_pi.py` (which validates PSNR/SSIM against benchmark datasets),
this script is for actual deployment use: feed it any low-resolution image
and it produces the super-resolved output.

Design choices that differ from test_pi.py:
  * No HR ground-truth required.
  * No PSNR/SSIM computation.
  * Supports a single image OR a directory of images.
  * Patch-wise inference (SFMformerModel.test()) is still used so output quality
    matches paper numbers.

Usage
-----
    python inference_pi.py -i input.png -o results/ --scale 2
    python inference_pi.py -i ./my_lr_images/ -o ./my_sr_results/ --scale 2
    python inference_pi.py -i input.png --scale 4
    python inference_pi.py -i input.png --scale 2 --time
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))


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


def img2tensor(img_pil: Image.Image) -> torch.Tensor:
    arr = np.array(img_pil.convert('RGB'))
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0)


def tensor2img(t: torch.Tensor) -> Image.Image:
    arr = t.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


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
    else:
        sd = raw

    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    print('  Strict load OK')


def process_image(model: torch.nn.Module, in_path: Path, out_path: Path,
                  scale: int, verbose: bool = True, timed: bool = False) -> None:
    img_lr = Image.open(in_path).convert('RGB')
    if verbose:
        print(f'  {in_path.name}  ({img_lr.size[0]}x{img_lr.size[1]})  -> ', end='', flush=True)

    lq = img2tensor(img_lr)
    t0 = time.perf_counter() if timed else None
    sr_tensor = patchwise_test(model, lq, scale=scale)
    dt = time.perf_counter() - t0 if timed else None

    img_sr = tensor2img(sr_tensor)
    img_sr.save(out_path)

    if verbose:
        print(f'  {img_sr.size[0]}x{img_sr.size[1]}', end='')
        if timed:
            print(f'  [{dt:.2f}s]')
        else:
            print()


SUPPORTED_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def main():
    ap = argparse.ArgumentParser(description='SFMformer single-image SR (Pi 5 CPU)')
    ap.add_argument('-i', '--in_path', type=str, required=True,
                    help='Input image file OR directory')
    ap.add_argument('-o', '--out_path', type=str, default='results/',
                    help='Output directory (created if missing). Default: results/')
    ap.add_argument('--scale', type=int, default=2, choices=[2, 3, 4],
                    help='Upscaling factor. Default: 2')
    ap.add_argument('--threads', type=int, default=4,
                    help='Torch CPU threads. Default: 4 (Pi 5 has 4 cores)')
    ap.add_argument('--time', action='store_true',
                    help='Print per-image inference time')
    ap.add_argument('--root', type=str, default=str(Path(__file__).parent),
                    help='Project root, where experiments/ lives')
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    in_path = Path(args.in_path)
    out_dir = Path(args.out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path(args.root)
    ckpt_path = root / 'experiments' / 'pretrained_models' / CHECKPOINT_MAP[args.scale]
    if not ckpt_path.exists():
        print(f'[ERROR] Checkpoint not found: {ckpt_path}')
        return
    if not in_path.exists():
        print(f'[ERROR] Input not found: {in_path}')
        return

    print(f'Building SFMformer x{args.scale}...')
    from basicsr.archs.sfmformer_arch import SFMformer
    model = SFMformer(upscale=args.scale, **SFMFORMER_LIGHT_CONFIG)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters: {n_params/1e3:.1f}K ({n_params/1e6:.2f}M)')

    print('Loading checkpoint...')
    load_state_dict_flexibly(model, str(ckpt_path))
    model.eval()

    if in_path.is_dir():
        files = sorted([p for p in in_path.iterdir()
                        if p.suffix.lower() in SUPPORTED_EXTS])
        print(f'\nProcessing {len(files)} image(s) from {in_path} ...')
    else:
        if in_path.suffix.lower() not in SUPPORTED_EXTS:
            print(f'[ERROR] Unsupported file type: {in_path.suffix}')
            return
        files = [in_path]
        print(f'\nProcessing 1 image: {in_path.name} ...')

    total_t0 = time.perf_counter()
    for f in files:
        out_name = f'{f.stem}_SFMformer_SRx{args.scale}{f.suffix}'
        out_path = out_dir / out_name
        process_image(model, f, out_path, scale=args.scale, timed=args.time)

    total_dt = time.perf_counter() - total_t0
    print(f'\nDone. {len(files)} image(s) in {total_dt:.2f}s '
          f'({total_dt/max(len(files),1):.2f}s/image avg)')
    print(f'Results saved to: {out_dir.absolute()}')


if __name__ == '__main__':
    main()