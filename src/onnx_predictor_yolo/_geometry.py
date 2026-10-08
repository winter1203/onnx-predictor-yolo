"""Image transforms and box geometry shared by all supported tasks."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray


def padding_color(value: int | tuple[int, int, int]) -> tuple[int, int, int]:
    """Validate a gray level or BGR color and return three integer channels."""
    channels = (value,) * 3 if isinstance(value, Integral) else value
    if (
        not isinstance(channels, tuple)
        or len(channels) != 3
        or any(
            isinstance(channel, bool)
            or not isinstance(channel, Integral)
            or not 0 <= channel <= 255
            for channel in channels
        )
    ):
        raise ValueError("pad_value must be an integer or a BGR tuple, with values in [0, 255].")
    return int(channels[0]), int(channels[1]), int(channels[2])


@dataclass(frozen=True)
class Transform:
    """Record the gain and actual integer padding applied to one image."""

    original_shape: tuple[int, int]
    input_shape: tuple[int, int]
    resized_shape: tuple[int, int]
    scale: tuple[float, float]
    padding: tuple[int, int]

    def restore_boxes(self, boxes: NDArray[np.float32]) -> NDArray[np.float32]:
        """Undo preprocessing for axis-aligned boxes and clip to the image."""
        restored = boxes.copy()
        restored[:, [0, 2]] = (restored[:, [0, 2]] - self.padding[0]) / self.scale[0]
        restored[:, [1, 3]] = (restored[:, [1, 3]] - self.padding[1]) / self.scale[1]
        return clip_boxes(restored, self.original_shape)

    def restore_obb(self, boxes: NDArray[np.float32]) -> NDArray[np.float32]:
        """Undo an isotropic transform without changing rectangle geometry."""
        if self.scale[0] != self.scale[1]:
            raise ValueError("Oriented boxes require letterbox preprocessing.")
        restored = boxes.copy()
        restored[:, :2] -= self.padding
        restored[:, :4] /= self.scale[0]
        return restored


def preprocess(
    image: NDArray[np.uint8],
    input_shape: tuple[int, int],
    dtype: type[np.float32] | type[np.float16],
    resize: Literal["letterbox", "stretch"],
    pad_value: int | tuple[int, int, int] = 114,
) -> tuple[NDArray, Transform]:
    """Convert BGR uint8 input to RGB NCHW, using a BGR padding color."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise TypeError("image must be a BGR uint8 NumPy array.")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) == 0:
        raise ValueError("image must have nonempty shape (height, width, 3).")
    height, width = image.shape[:2]
    target_h, target_w = input_shape
    if resize == "letterbox":
        gain = min(target_h / height, target_w / width)
        new_h = max(1, min(target_h, round(height * gain)))
        new_w = max(1, min(target_w, round(width * gain)))
        scale = (gain, gain)
    else:
        new_h, new_w = target_h, target_w
        scale = (target_w / width, target_h / height)
    left, top = (target_w - new_w) // 2, (target_h - new_h) // 2
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    canvas[top : top + new_h, left : left + new_w] = resized
    tensor = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=dtype)
    tensor /= dtype(255)
    transform = Transform((height, width), input_shape, (new_h, new_w), scale, (left, top))
    return tensor, transform


def xywh_to_xyxy(boxes: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert center coordinates and dimensions to corner coordinates."""
    half_size = boxes[:, 2:4] * 0.5
    return np.concatenate((boxes[:, :2] - half_size, boxes[:, :2] + half_size), axis=1)


def clip_boxes(
    boxes: NDArray[np.float32], shape: tuple[int, int]
) -> NDArray[np.float32]:
    """Clip axis-aligned boxes in place to continuous image boundaries."""
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])
    return boxes


def obb_to_corners(boxes: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert center/size/angle rectangles to four cyclic corner points."""
    angles = boxes[:, 4]
    axis_x = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    axis_y = np.stack((-np.sin(angles), np.cos(angles)), axis=1)
    horizontal = axis_x * boxes[:, 2:3] * 0.5
    vertical = axis_y * boxes[:, 3:4] * 0.5
    center = boxes[:, :2]
    return np.stack(
        (
            center - horizontal - vertical,
            center + horizontal - vertical,
            center + horizontal + vertical,
            center - horizontal + vertical,
        ),
        axis=1,
    ).astype(np.float32, copy=False)
