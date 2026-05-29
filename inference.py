"""
inference.py
============
Single-image super-resolution inference for SFMformer (lightweight).
Auto-detects GPU; falls back to CPU if no CUDA is available.

For Raspberry Pi / strictly CPU-only deployment with patch-wise inference,
use `inference_pi.py` instead.
"""
import os
import os.path as osp
import argparse

import torch
from PIL import Image
from torchvision import transforms

from basicsr.archs.sfmformer_arch import SFMformer


MODEL_PATH = {
    2: "experiments/pretrained_models/101_SFMformer_SRx2_scratch.pth",
    3: "experiments/pretrained_models/102_SFMformer_SRx3_finetune.pth",
    4: "experiments/pretrained_models/103_SFMformer_SRx4_finetune.pth",
}


def get_parser():
    parser = argparse.ArgumentParser(description="SFMformer single-image SR")
    parser.add_argument("-i", "--in_path", type=str, required=True,
                        help="Input image or directory path.")
    parser.add_argument("-o", "--out_path", type=str, default="results/test/",
                        help="Output directory. Default: results/test/")
    parser.add_argument("--scale", type=int, default=4, choices=[2, 3, 4],
                        help="Scale factor for SR. Default: 4")
    return parser.parse_args()


def process_image(image_input_path, image_output_path, model, device):
    with torch.no_grad():
        image_input = Image.open(image_input_path).convert('RGB')
        image_input = transforms.ToTensor()(image_input).unsqueeze(0).to(device)
        image_output = model(image_input).clamp(0.0, 1.0)[0].cpu()
        image_output = transforms.ToPILImage()(image_output)
        image_output.save(image_output_path)


def main():
    args = get_parser()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # SFMformer-light config (matches options/test/101_SFMformer_SRx2_scratch.yml)
    model = SFMformer(
        upscale=args.scale,
        embed_dim=52,
        depths=[2, 4, 6, 6, 6],
        num_heads=4,
        num_topk=[1024, 1024,
                  256, 256, 256, 256,
                  128, 128, 128, 128, 128, 128,
                  64, 64, 64, 64, 64, 64,
                  32, 32, 32, 32, 32, 32],
        window_size=32,
        convffn_kernel_size=7,
        mlp_ratio=1,
        upsampler='pixelshuffledirect',
        use_checkpoint=False,
    )

    ckpt_path = MODEL_PATH[args.scale]
    print(f"Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)['params_ema']
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    os.makedirs(args.out_path, exist_ok=True)

    def _run(in_file, out_dir):
        file_name, ext = osp.splitext(osp.basename(in_file))
        out_file = osp.join(out_dir,
                            f"{file_name}_SFMformer_SRx{args.scale}{ext}")
        process_image(in_file, out_file, model, device)
        print(f"  {osp.basename(in_file)} -> {osp.basename(out_file)}")

    if osp.isdir(args.in_path):
        files = [f for f in os.listdir(args.in_path)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Processing {len(files)} image(s) from {args.in_path} ...")
        for file in files:
            _run(osp.join(args.in_path, file), args.out_path)
    else:
        if args.in_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            _run(args.in_path, args.out_path)
        else:
            print(f"[ERROR] Unsupported file type: {args.in_path}")
            return

    print(f"\nDone. Results saved to: {osp.abspath(args.out_path)}")


if __name__ == "__main__":
    main()