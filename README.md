# SFMformer-SR Deploy

Deployment of **SFMformer** — a lightweight (0.97M-parameter) Transformer for single-image super-resolution. Runs on a GPU (Windows/Linux) or CPU-only on edge devices such as the Raspberry Pi 5; the same code auto-selects the device.

## Benchmark Results

PSNR (dB) ↑ / SSIM ↑ on the standard SR test sets (Y-channel), at ×2 / ×3 / ×4.

| Dataset  | ×2              | ×3              | ×4              |
|----------|-----------------|-----------------|-----------------|
| Set5     | 38.40 / 0.9620  | 34.88 / 0.9311  | 32.70 / 0.9009  |
| Set14    | 34.20 / 0.9227  | 30.79 / 0.8500  | 28.98 / 0.7900  |
| BSD100   | 32.45 / 0.9032  | 29.38 / 0.8125  | 27.82 / 0.7450  |
| Urban100 | 33.53 / 0.9397  | 29.37 / 0.8744  | 27.13 / 0.8158  |
| Manga109 | 39.73 / 0.9794  | 34.88 / 0.9516  | 31.65 / 0.9212  |

## Install

PyTorch first — pick the build for your machine:

```bash
# GPU (Windows / Linux with an NVIDIA card)
pip install torch torchvision

# Raspberry Pi / CPU-only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then the rest:

```bash
pip install -r requirements.txt
```

## Models
You can downlaod pretrained models in https://drive.google.com/drive/folders/13xxMaTwbKoBIJVuUaaGjb1naCsxnM8-k?usp=sharing

## Usage

### Inference (just run)

GUI — load an image, pick a scale, run:

```bash
python sfm_gui.py
```

Command line, single image or a folder:

```bash
python inference_pi.py -i input.png --scale 4 -o output/result.png
python inference_pi.py -i ./my_images/ --scale 2
```

### Benchmark (PSNR / SSIM)

To reproduce the table you first need the test sets **in place**. Download Set5 / Set14 / BSD100 / Urban100 / Manga109 from the [HiT-SR repo](https://github.com/XiangZ-0/HiT-SR) and put them under `datasets/`:

```
datasets/TestDataSR/
├── HR/{Set5,Set14,BSD100,Urban100,Manga109}/{x2,x3,x4}/
└── LR/LRBI/{Set5,Set14,BSD100,Urban100,Manga109}/{x2,x3,x4}/
```

Then run:

```bash
python test_pi.py --scale 2 --benchmark Set5
python test_pi.py --scale 4 --benchmark Urban100
```

(The GUI also has a built-in **Benchmark** panel that does the same once the datasets are in place.)

## Repository Structure

```
sfmformer-sr-deploy/
├── sfm_gui.py                      # interactive GUI (inference + benchmark)
├── inference_pi.py                 # CLI inference (single image / folder)
├── test_pi.py                      # CLI PSNR/SSIM benchmark
├── basicsr/archs/
│   ├── sfmformer_arch.py           # model definition
│   ├── sfmformer_cpu_ops.py        # CPU fallbacks for the custom ops
│   ├── idynamicdwconv_util.py      # dynamic depth-wise conv
│   └── arch_util.py
├── experiments/pretrained_models/  # 101_x2 / 102_x3 / 103_x4 .pth
├── requirements.txt
└── LICENSE.txt
```

## Citation

```bibtex
@inproceedings{sfmformer,
  title={...},
  author={...},
  booktitle={...},
  year={2026}
}
```

Benchmark test sets (Set5, Set14, BSD100, Urban100, Manga109) are obtained from the [HiT-SR](https://github.com/XiangZ-0/HiT-SR) repository.

## License

Apache-2.0 — see `LICENSE.txt`.