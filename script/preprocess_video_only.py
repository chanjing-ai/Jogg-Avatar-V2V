#!/usr/bin/env python3
"""
单独运行视频预处理：将输入视频转为 .tensors.vae2.2.pth，供推理使用。
必须提供 _mouth_info.json（含 face_bbox），无人脸检测。

Usage:
  uv run python script/preprocess_video_only.py --video_path input.mp4 \\
    --vae_path models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \\
    --mouth_info_path input_mouth_info.json
"""
import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def main():
    parser = argparse.ArgumentParser(description="Preprocess video to VAE features (.tensors.vae2.2.pth), requires mouth_info json")
    parser.add_argument("--video_path", type=str, required=True, help="Input video path (e.g. .mp4)")
    parser.add_argument("--vae_path", type=str, required=True, help="Path to Wan2.2_VAE.pth")
    parser.add_argument("--mouth_info_path", type=str, required=True, help="Path to _mouth_info.json (must contain face_bbox)")
    parser.add_argument("--output_path", type=str, default=None, help="Output .pth path (default: video_path + .tensors.vae2.2.pth)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for VAE")
    parser.add_argument("--height", type=int, default=480, help="Face crop height (default: 480)")
    parser.add_argument("--width", type=int, default=480, help="Face crop width (default: 480)")
    args = parser.parse_args()

    from Avatar.utils.video_preprocess import preprocess_video_to_vae_features

    out = preprocess_video_to_vae_features(
        args.video_path,
        vae_path=args.vae_path,
        mouth_info_path=args.mouth_info_path,
        output_path=args.output_path,
        device=args.device,
        height=args.height,
        width=args.width,
    )
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
