# SFMformer-SR Deploy

Deployment of **SFMformer** — a lightweight (0.97M-parameter) Transformer for single-image super-resolution. Runs on a GPU (Windows/Linux) or CPU-only on edge devices such as the Raspberry Pi 5; the same code auto-selects the device.

This repository is the inference / deployment interface for the main SFMformer project: JoeyYCH/sfmformer. See that repo for the training code, full benchmark results, and the paper.


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
  year={2025}
}
```

Benchmark test sets (Set5, Set14, BSD100, Urban100, Manga109) are obtained from the [HiT-SR](https://github.com/XiangZ-0/HiT-SR) repository.

## License

Apache-2.0 — see `LICENSE.txt`.