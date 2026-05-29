# SFMformer-SR Deploy

End-to-end deployment of **SFMformer** — a lightweight Transformer-based single-image super-resolution model — on resource-constrained edge devices such as the Raspberry Pi 5.

## Highlights

- **0.97M parameters** lightweight design achieving SOTA-level PSNR/SSIM on standard benchmarks
- **CPU-only inference**: pure-PyTorch fallbacks for custom CUDA kernels (SMM, IDynamic), enabling deployment on devices without GPU
- **Bit-equivalent numerical accuracy**: PSNR/SSIM on Pi 5 CPU matches the GPU implementation within 0.005 dB (verified on Set5/Set14 ×2/×3/×4)
- **Reproducible**: numerical sanity checks (`test_cpu_ops.py`) verify the CPU fallback against reference implementations and autograd gradients (fp64 machine precision)

## Results — Pi 5 CPU vs. Paper (GPU)

| Dataset | Scale | Paper PSNR | Pi 5 PSNR | Paper SSIM | Pi 5 SSIM |
|---------|-------|-----------:|----------:|-----------:|----------:|
| Set5    | ×2    | 38.42      | **38.4226** | 0.9621   | **0.9621**  |
| Set5    | ×3    | 34.87      | **34.8719** | 0.9311   | **0.9311**  |
| Set5    | ×4    | 32.68      | **32.6751** | 0.9007   | **0.9007**  |
| Set14   | ×2    | 34.19      | **34.1919** | 0.9225   | **0.9225**  |
| Set14   | ×3    | 30.78      | **30.7833** | 0.8499   | **0.8499**  |
| Set14   | ×4    | 29.02      | **29.0220** | 0.7906   | **0.7906**  |

All 6 benchmarks aligned within < 0.005 dB / < 0.0005 SSIM error.

## Repository Structure

```
sfmformer-sr-deploy/
├── basicsr/
│   ├── archs/
│   │   ├── sfmformer_arch.py       # SFMformer model definition
│   │   ├── sfmformer_cpu_ops.py    # CPU fallbacks for SMM / IDynamic
│   │   ├── idynamicdwconv_util.py  # IDynamic depthwise conv (CuPy-guarded)
│   │   ├── arch_util.py
│   │   └── test_cpu_ops.py         # Numerical sanity check (fp64 gradcheck)
│   ├── data/   losses/   metrics/   models/   utils/   # Training-side code
│   ├── train.py    test.py    version.py
│
├── experiments/
│   └── pretrained_models/
│       ├── 101_SFMformer_SRx2_scratch.pth
│       ├── 102_SFMformer_SRx3_finetune.pth
│       └── 103_SFMformer_SRx4_finetune.pth
│
├── ops_smm/                # Original CUDA kernel sources (build-from-source on GPU)
├── options/                # Training / testing yaml configs
│
├── inference.py            # Original GPU inference (single image)
├── inference_pi.py         # CPU inference for Pi 5 (single image)
├── test_pi.py              # PSNR/SSIM benchmark validation on Pi 5
├── sr_visual_compare.py    # Visual comparison tool
├── setup.py                # Optional: install as package
├── requirements.txt
└── LICENSE.txt
```

## Quick Start (Raspberry Pi 5)

### 1. Environment

- Raspberry Pi 5 (4-core ARM Cortex-A76, 8 GB+ RAM recommended)
- Debian 13 (Trixie) or Raspberry Pi OS Bookworm+
- Python 3.11+
- PyTorch CPU build (no CUDA needed)

```bash
python -m venv ~/pft_infer
source ~/pft_infer/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Single-image super-resolution

```bash
python inference_pi.py -i input.png --scale 2
python inference_pi.py -i input.png --scale 4 -o output/result.png
python inference_pi.py -i ./my_images/ --scale 2 --time   # batch + timing
```

### 3. PSNR/SSIM validation against the paper

```bash
python test_pi.py --scale 2 --benchmark Set5
python test_pi.py --scale 3 --benchmark Set14
python test_pi.py --scale 4 --benchmark Set5
```

Expects benchmark datasets under:

```
datasets/TestDataSR/
├── HR/{Set5,Set14,...}/{x2,x3,x4}/
└── LR/LRBI/{Set5,Set14,...}/{x2,x3,x4}/
```

Match within 0.01 dB / 0.0005 SSIM confirms deployment correctness.

### 4. Verify CPU fallback numerical accuracy

```bash
cd basicsr/archs
python test_cpu_ops.py
```

Should print `All tests passed ✅` with `max abs error ≈ 1e-15` (fp64 machine precision) and `gradcheck: True` for both forward and backward.

## How the CPU Fallback Works

The model uses three custom CUDA operations during training:

| CUDA op | Purpose | CPU equivalent in `sfmformer_cpu_ops.py` |
|---|---|---|
| `smm_cuda.SMM_QmK_forward_cuda` | Top-k sparse attention scores | `smm_qmk()` via `torch.matmul + torch.gather` with query chunking |
| `smm_cuda.SMM_AmV_forward_cuda` | Top-k sparse attention values | `smm_amv()` via gather + weighted sum, query-chunked |
| `idynamicdwconv_util._idynamic_cuda` | Per-pixel dynamic depthwise conv | `idynamic_conv()` via padded indexing + broadcasting |

Forward and backward are handled automatically by PyTorch autograd — no hand-written gradient code. The fallback uses identical math, so trained CUDA weights load directly into the CPU model without retraining.

Numerical equivalence is verified to fp64 machine precision (~7e-15) via `test_cpu_ops.py`.

## Citation

If you use this codebase, please cite the SFMformer paper:

```bibtex
@inproceedings{sfmformer,
  title={...},
  author={...},
  booktitle={...},
  year={2025}
}
```

## License

Apache-2.0 — see `LICENSE.txt`.
