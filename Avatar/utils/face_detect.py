# 从 _mouth_info.json 读取人脸框（必须提供，无人脸检测）

from __future__ import annotations

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from statistics import median


def _resolve_mouth_info_path(video_path: str, mouth_info_path: Optional[str] = None) -> str:
    if mouth_info_path is None:
        return video_path.rsplit(".", 1)[0] + "_mouth_info.json"
    return mouth_info_path


def _load_mouth_info_data(video_path: str, mouth_info_path: Optional[str] = None) -> Optional[dict]:
    path = _resolve_mouth_info_path(video_path, mouth_info_path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_face_bbox_from_mouth_info(
    video_path: str,
    mouth_info_path: Optional[str] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """
    从 _mouth_info.json 读取第一帧的人脸框。
    返回 (xmin, xmax, ymin, ymax) 或 None。
    """
    data = _load_mouth_info_data(video_path, mouth_info_path)
    if data is None:
        return None
    first = data.get("1") or data.get(1)
    if not first:
        return None
    bbox = first.get("face_bbox")
    if not bbox or len(bbox) < 4:
        return None
    return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))


def load_all_face_bboxes(
    video_path: str,
    mouth_info_path: Optional[str] = None,
) -> Optional[List[Tuple[int, int, int, int]]]:
    """
    从 _mouth_info.json 加载所有帧的 face_bbox，按帧序排列（0-based list）。
    返回 [(xmin, xmax, ymin, ymax), ...] 或 None。
    """
    data = _load_mouth_info_data(video_path, mouth_info_path)
    if data is None:
        return None
    sorted_keys = sorted(data.keys(), key=lambda k: int(k))
    result = []
    for key in sorted_keys:
        bbox = data[key].get("face_bbox")
        if bbox and len(bbox) >= 4:
            result.append((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
    return result if result else None


def smooth_bboxes(
    bboxes: List[Tuple[int, int, int, int]],
    window: int = 7,
) -> List[Tuple[int, int, int, int]]:
    """
    对逐帧 bbox 做滑动窗口平滑，减少抖动。
    window 为平滑窗口大小（奇数），默认 7。
    """
    n = len(bboxes)
    if n <= 1:
        return list(bboxes)
    half = window // 2
    smoothed = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = bboxes[lo:hi]
        avg = tuple(int(round(sum(b[c] for b in chunk) / len(chunk))) for c in range(4))
        smoothed.append(avg)
    return smoothed


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    return ordered[idx]


def _bbox_to_components(bbox: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
    x1, x2, y1, y2 = [float(v) for v in bbox]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    return cx, cy, w, h


def _components_to_bbox(cx: float, cy: float, w: float, h: float) -> Tuple[int, int, int, int]:
    return (
        int(round(cx - w / 2.0)),
        int(round(cx + w / 2.0)),
        int(round(cy - h / 2.0)),
        int(round(cy + h / 2.0)),
    )


def _stabilize_bboxes_once(
    bboxes: List[Tuple[int, int, int, int]],
    history: int,
    center_ratio: float,
    shrink_floor_ratio: float,
    grow_cap_ratio: float,
    max_center_step: float,
    max_size_step: float,
    floor_quantile: float,
) -> List[Tuple[int, int, int, int]]:
    stabilized: List[Tuple[int, int, int, int]] = []
    for bbox in bboxes:
        cx, cy, w, h = _bbox_to_components(bbox)

        if stabilized:
            hist = stabilized[max(0, len(stabilized) - history):]
            hist_cx = [(_bbox_to_components(prev)[0]) for prev in hist]
            hist_cy = [(_bbox_to_components(prev)[1]) for prev in hist]
            hist_w = [(_bbox_to_components(prev)[2]) for prev in hist]
            hist_h = [(_bbox_to_components(prev)[3]) for prev in hist]

            ref_cx = median(hist_cx)
            ref_cy = median(hist_cy)
            ref_w = _percentile(hist_w, floor_quantile)
            ref_h = _percentile(hist_h, floor_quantile)

            max_dx = max(max_center_step, ref_w * center_ratio)
            max_dy = max(max_center_step, ref_h * center_ratio)
            near_same_track = (
                abs(cx - ref_cx) <= max_dx * 1.5 and
                abs(cy - ref_cy) <= max_dy * 1.5
            )

            if near_same_track:
                cx = _clamp(cx, ref_cx - max_dx, ref_cx + max_dx)
                cy = _clamp(cy, ref_cy - max_dy, ref_cy + max_dy)
                w = _clamp(w, ref_w * shrink_floor_ratio, ref_w * grow_cap_ratio)
                h = _clamp(h, ref_h * shrink_floor_ratio, ref_h * grow_cap_ratio)

            prev_cx, prev_cy, prev_w, prev_h = _bbox_to_components(stabilized[-1])
            cx = _clamp(cx, prev_cx - max_center_step, prev_cx + max_center_step)
            cy = _clamp(cy, prev_cy - max_center_step, prev_cy + max_center_step)
            w = _clamp(w, prev_w - max_size_step, prev_w + max_size_step)
            h = _clamp(h, prev_h - max_size_step, prev_h + max_size_step)

        stabilized.append(_components_to_bbox(cx, cy, w, h))

    return stabilized


def _freeze_bbox_if_motion_small(
    bboxes: List[Tuple[int, int, int, int]],
    center_motion_ratio: float,
    min_center_motion_px: float,
    size_quantile: float,
) -> List[Tuple[int, int, int, int]]:
    if len(bboxes) <= 1:
        return list(bboxes)

    components = [_bbox_to_components(bbox) for bbox in bboxes]
    centers_x = [item[0] for item in components]
    centers_y = [item[1] for item in components]
    widths = [item[2] for item in components]
    heights = [item[3] for item in components]

    frozen_w = int(round(_percentile(widths, size_quantile)))
    frozen_h = int(round(_percentile(heights, size_quantile)))
    max_x_range = max(min_center_motion_px, frozen_w * center_motion_ratio)
    max_y_range = max(min_center_motion_px, frozen_h * center_motion_ratio)
    x_range = max(centers_x) - min(centers_x)
    y_range = max(centers_y) - min(centers_y)

    if x_range > max_x_range or y_range > max_y_range:
        return bboxes

    frozen_cx = median(centers_x)
    frozen_cy = median(centers_y)
    frozen_bbox = _components_to_bbox(frozen_cx, frozen_cy, frozen_w, frozen_h)
    return [frozen_bbox] * len(bboxes)


def stabilize_bboxes(
    bboxes: List[Tuple[int, int, int, int]],
    history: int = 65,
    center_ratio: float = 0.035,
    shrink_floor_ratio: float = 0.998,
    grow_cap_ratio: float = 1.01,
    max_center_step: float = 1.5,
    max_size_step: float = 0.8,
    floor_quantile: float = 0.95,
    freeze_size_motion_ratio: float = 0.10,
    freeze_size_min_motion_px: float = 16.0,
    freeze_size_quantile: float = 0.90,
) -> List[Tuple[int, int, int, int]]:
    """
    在滑动均值之后，再做一层时序稳定：
    - 中心点限制在近期轨迹附近，避免短时跳动
    - 宽高限制在近期中位数附近，避免 SCRFD 突然切到过紧的人脸框
    """
    if len(bboxes) <= 1:
        return list(bboxes)

    forward = _stabilize_bboxes_once(
        bboxes,
        history=history,
        center_ratio=center_ratio,
        shrink_floor_ratio=shrink_floor_ratio,
        grow_cap_ratio=grow_cap_ratio,
        max_center_step=max_center_step,
        max_size_step=max_size_step,
        floor_quantile=floor_quantile,
    )
    backward = list(reversed(_stabilize_bboxes_once(
        list(reversed(forward)),
        history=history,
        center_ratio=center_ratio,
        shrink_floor_ratio=shrink_floor_ratio,
        grow_cap_ratio=grow_cap_ratio,
        max_center_step=max_center_step,
        max_size_step=max_size_step,
        floor_quantile=floor_quantile,
    )))
    return _freeze_bbox_if_motion_small(
        backward,
        center_motion_ratio=freeze_size_motion_ratio,
        min_center_motion_px=freeze_size_min_motion_px,
        size_quantile=freeze_size_quantile,
    )


def get_face_bbox_required(
    video_path: str,
    mouth_info_path: str,
) -> Tuple[int, int, int, int]:
    """
    从 mouth_info_path 读取第一帧人脸框；必须存在且有效，否则抛错。
    返回 (xmin, xmax, ymin, ymax)。
    """
    bbox = load_face_bbox_from_mouth_info(video_path, mouth_info_path)
    if bbox is None:
        path = mouth_info_path or (video_path.rsplit(".", 1)[0] + "_mouth_info.json")
        raise FileNotFoundError(
            f"mouth_info 必须提供且有效。请提供 _mouth_info.json 路径（含 face_bbox）：{path}"
        )
    return bbox


def get_all_face_bboxes_required(
    video_path: str,
    mouth_info_path: str,
    smooth_window: int = 7,
) -> List[Tuple[int, int, int, int]]:
    """
    加载所有帧的 face_bbox，做平滑后返回。
    """
    bboxes = load_all_face_bboxes(video_path, mouth_info_path)
    if bboxes is None:
        path = mouth_info_path or (video_path.rsplit(".", 1)[0] + "_mouth_info.json")
        raise FileNotFoundError(
            f"mouth_info 必须提供且有效。请提供 _mouth_info.json 路径（含 face_bbox）：{path}"
        )
    if smooth_window > 1:
        bboxes = smooth_bboxes(bboxes, smooth_window)
    bboxes = stabilize_bboxes(bboxes)
    return bboxes


def _distance2bbox(points, distance):
    return np.stack(
        [
            points[:, 0] - distance[:, 0],
            points[:, 1] - distance[:, 1],
            points[:, 0] + distance[:, 2],
            points[:, 1] + distance[:, 3],
        ],
        axis=-1,
    )


def _distance2kps(points, distance):
    predictions = []
    for index in range(0, distance.shape[1], 2):
        predictions.extend(
            [points[:, 0] + distance[:, index], points[:, 1] + distance[:, index + 1]]
        )
    return np.stack(predictions, axis=-1)


class SCRFDDetector:
    """Minimal SCRFD ONNX detector used by the public preprocessing CLI."""

    def __init__(self, model_path: str, provider: str = "gpu"):
        import cv2
        import numpy as np
        import onnxruntime

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"SCRFD model not found: {model_path}")

        available = set(onnxruntime.get_available_providers())
        providers = ["CPUExecutionProvider"]
        if provider == "gpu" and "CUDAExecutionProvider" in available:
            providers.insert(0, "CUDAExecutionProvider")
        self.cv2 = cv2
        self.np = np
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.center_cache = {}

        output_count = len(self.output_names)
        if output_count in (6, 9):
            self.strides = [8, 16, 32]
            self.num_anchors = 2
        elif output_count in (10, 15):
            self.strides = [8, 16, 32, 64, 128]
            self.num_anchors = 1
        else:
            raise ValueError(f"Unsupported SCRFD output count: {output_count}")
        self.fmc = len(self.strides)
        self.use_kps = output_count in (9, 15)

    def _anchors(self, height: int, width: int, stride: int):
        key = (height, width, stride)
        if key not in self.center_cache:
            centers = self.np.stack(
                self.np.mgrid[:height, :width][::-1], axis=-1
            ).astype(self.np.float32)
            centers = (centers * stride).reshape((-1, 2))
            if self.num_anchors > 1:
                centers = self.np.repeat(centers, self.num_anchors, axis=0)
            self.center_cache[key] = centers
        return self.center_cache[key]

    @staticmethod
    def _nms(detections, threshold: float):
        import numpy as np

        x1, y1, x2, y2 = (detections[:, index] for index in range(4))
        scores = detections[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size:
            index = order[0]
            keep.append(index)
            xx1 = np.maximum(x1[index], x1[order[1:]])
            yy1 = np.maximum(y1[index], y1[order[1:]])
            xx2 = np.minimum(x2[index], x2[order[1:]])
            yy2 = np.minimum(y2[index], y2[order[1:]])
            intersection = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(
                0.0, yy2 - yy1 + 1
            )
            overlap = intersection / (areas[index] + areas[order[1:]] - intersection)
            order = order[np.where(overlap <= threshold)[0] + 1]
        return keep

    def detect(
        self,
        frame,
        threshold: float = 0.5,
        input_size: Tuple[int, int] = (640, 640),
        nms_threshold: float = 0.4,
    ):
        np = self.np
        cv2 = self.cv2
        frame_ratio = frame.shape[0] / frame.shape[1]
        model_ratio = input_size[1] / input_size[0]
        if frame_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / frame_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * frame_ratio)
        scale = new_height / frame.shape[0]
        resized = cv2.resize(frame, (new_width, new_height))
        detector_input = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        detector_input[:new_height, :new_width] = resized
        blob = cv2.dnn.blobFromImage(
            detector_input,
            1.0 / 128,
            input_size,
            (127.5, 127.5, 127.5),
            swapRB=True,
        )
        outputs = self.session.run(self.output_names, {self.input_name: blob})

        scores_list, boxes_list, keypoints_list = [], [], []
        for index, stride in enumerate(self.strides):
            scores = outputs[index]
            boxes = outputs[index + self.fmc] * stride
            keypoints = outputs[index + self.fmc * 2] * stride if self.use_kps else None
            if scores.ndim == 3:
                scores, boxes = scores[0], boxes[0]
                if keypoints is not None:
                    keypoints = keypoints[0]

            height, width = blob.shape[2] // stride, blob.shape[3] // stride
            anchors = self._anchors(height, width, stride)
            positive = np.where(scores.reshape(-1) >= threshold)[0]
            scores_list.append(scores.reshape(-1)[positive])
            boxes_list.append(_distance2bbox(anchors, boxes)[positive])
            if keypoints is not None:
                points = _distance2kps(anchors, keypoints).reshape((-1, 5, 2))
                keypoints_list.append(points[positive])

        if not any(len(scores) for scores in scores_list):
            return None, None
        scores = np.concatenate(scores_list)
        boxes = np.vstack(boxes_list) / scale
        order = scores.argsort()[::-1]
        detections = np.hstack([boxes, scores[:, None]])[order].astype(np.float32)
        keep = self._nms(detections, nms_threshold)
        detections = detections[keep]
        keypoints = None
        if self.use_kps:
            keypoints = np.vstack(keypoints_list)[order][keep] / scale

        areas = (detections[:, 2] - detections[:, 0]) * (
            detections[:, 3] - detections[:, 1]
        )
        best = int(areas.argmax())
        return detections[best], keypoints[best] if keypoints is not None else None

    def face_crop_bbox(self, frame) -> Optional[Tuple[int, int, int, int]]:
        detection, keypoints = self.detect(frame)
        if detection is None:
            return None
        x1, y1, x2, y2 = detection[:4]
        width = x2 - x1
        center_x = x1 + width / 2
        eye_y = (keypoints[0][1] + keypoints[1][1]) / 2 if keypoints is not None else y1 + width * 0.35
        crop_x1 = int(center_x - width * 0.72)
        crop_x2 = int(center_x + width * 0.72)
        crop_y1 = int(eye_y - width * 1.44 * 0.2)
        crop_y2 = int(eye_y + width * 1.44 * 0.8)
        if crop_x2 - crop_x1 < 50 or crop_y2 - crop_y1 < 50:
            return None
        return crop_x1, crop_x2, crop_y1, crop_y2
