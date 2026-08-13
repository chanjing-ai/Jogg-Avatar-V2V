#!/usr/bin/env python3
"""Extract stabilized per-frame face boxes for Wan2.2 V2V preprocessing."""

import argparse
import json
import os
import sys

import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Avatar.utils.face_detect import SCRFDDetector, smooth_bboxes, stabilize_bboxes


def main():
    parser = argparse.ArgumentParser(
        description="Extract SCRFD face boxes to <video>_mouth_info.json."
    )
    parser.add_argument("--video_path", required=True, help="Input video path.")
    parser.add_argument("--scrfd_model", required=True, help="SCRFD ONNX model path.")
    parser.add_argument("--output_path", help="Output JSON path.")
    parser.add_argument(
        "--provider",
        choices=("gpu", "cpu"),
        default="gpu",
        help="Preferred ONNX Runtime provider.",
    )
    parser.add_argument("--smooth_window", type=int, default=7)
    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        raise FileNotFoundError(f"Video not found: {args.video_path}")

    detector = SCRFDDetector(args.scrfd_model, provider=args.provider)
    capture = cv2.VideoCapture(args.video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video_path}")

    boxes = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            boxes.append(detector.face_crop_bbox(frame))
    finally:
        capture.release()

    detected = [box for box in boxes if box is not None]
    if not detected:
        raise RuntimeError(f"No face detected in video: {args.video_path}")

    first_box = detected[0]
    filled_boxes = []
    previous = first_box
    for box in boxes:
        if box is not None:
            previous = box
        filled_boxes.append(previous)

    if args.smooth_window > 1:
        filled_boxes = smooth_bboxes(filled_boxes, args.smooth_window)
    filled_boxes = stabilize_bboxes(filled_boxes)

    output_path = args.output_path or (
        os.path.splitext(args.video_path)[0] + "_mouth_info.json"
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    data = {
        str(index): {"face_bbox": [int(value) for value in box]}
        for index, box in enumerate(filled_boxes, start=1)
    }
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
    print(f"Saved {len(data)} face boxes to {output_path}")


if __name__ == "__main__":
    main()
