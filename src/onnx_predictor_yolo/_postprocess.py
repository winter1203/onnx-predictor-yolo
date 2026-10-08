"""Decode export protocols independently of model names and file paths."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from ._geometry import Transform, clip_boxes, obb_to_corners, xywh_to_xyxy
from .results import Results

Task = Literal["detect", "segment", "obb"]
OutputFormat = Literal["auto", "raw", "end2end"]
Layout = Literal["channels_first", "channels_last"]


def prediction_rows(
    output: NDArray,
    extra_channels: int,
    output_format: OutputFormat,
    layout: Layout,
    num_classes: int | None,
) -> tuple[NDArray[np.float32], bool]:
    """Resolve raw versus processed output; reject ambiguous auto detection."""
    if output.ndim != 3 or output.shape[0] != 1:
        raise ValueError("Predictions must have rank 3 and batch size 1.")
    matrix = output[0].astype(np.float32, copy=False)
    raw = matrix.T if layout == "channels_first" else matrix
    raw_classes = raw.shape[1] - 4 - extra_channels
    raw_valid = raw_classes > 0 and (num_classes is None or raw_classes == num_classes)
    processed_valid = matrix.shape[1] == 6 + extra_channels
    if output_format == "auto":
        if raw_valid == processed_valid:
            raise ValueError(
                f"Cannot unambiguously decode prediction shape {output.shape}. "
                "Set output_format='raw' or 'end2end'; for raw outputs also check "
                "layout and num_classes (or names)."
            )
        output_format = "raw" if raw_valid else "end2end"
    if output_format == "raw":
        if not raw_valid:
            raise ValueError("Raw output channels do not match the task and class count.")
        return raw, False
    if not processed_valid:
        raise ValueError(f"End-to-end predictions must have {6 + extra_channels} columns.")
    return matrix, True


def non_max_suppression(
    boxes: NDArray[np.float32],
    scores: NDArray[np.float32],
    class_ids: NDArray[np.int64],
    iou: float,
    max_det: int,
    agnostic: bool,
    rotated: bool,
) -> NDArray[np.int64]:
    """Apply OpenCV NMS per class, with genuine rotated IoU for OBB."""
    groups = np.zeros_like(class_ids) if agnostic else class_ids
    kept: list[int] = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        candidates = boxes[indices]
        if rotated:
            rectangles = [
                ((float(x), float(y)), (float(w), float(h)), float(np.degrees(angle)))
                for x, y, w, h, angle in candidates
            ]
            selected = cv2.dnn.NMSBoxesRotated(rectangles, scores[indices].tolist(), 0.0, iou)
        else:
            rectangles = candidates.copy()
            rectangles[:, 2:] -= rectangles[:, :2]
            selected = cv2.dnn.NMSBoxes(rectangles.tolist(), scores[indices].tolist(), 0.0, iou)
        local = np.asarray(selected, dtype=np.int64).reshape(-1)
        kept.extend(indices[local].tolist())
    keep = np.asarray(kept, dtype=np.int64)
    return keep[np.argsort(-scores[keep], kind="stable")[:max_det]]


def decode_masks(
    prototypes: NDArray[np.float32],
    coefficients: NDArray[np.float32],
    boxes: NDArray[np.float32],
    transform: Transform,
    threshold: float,
) -> NDArray[np.bool_]:
    """Restore mask logits, remove padding, and crop in original coordinates."""
    height, width = transform.original_shape
    masks = np.zeros((len(boxes), height, width), dtype=np.bool_)
    if not len(boxes):
        return masks
    channels, proto_h, proto_w = prototypes.shape
    logits = coefficients @ prototypes.reshape(channels, -1)
    logits = logits.reshape(-1, proto_h, proto_w)
    logit_threshold = np.log(threshold / (1.0 - threshold))
    input_h, input_w = transform.input_shape
    resized_h, resized_w = transform.resized_shape
    left, top = transform.padding
    for index, (logit, box) in enumerate(zip(logits, boxes, strict=True)):
        expanded = cv2.resize(logit, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        unpadded = expanded[top : top + resized_h, left : left + resized_w]
        restored = cv2.resize(unpadded, (width, height), interpolation=cv2.INTER_LINEAR)
        x1, y1, x2, y2 = np.ceil(box).astype(np.int64)
        masks[index, y1:y2, x1:x2] = restored[y1:y2, x1:x2] > logit_threshold
    return masks


def postprocess(
    output: NDArray,
    prototypes: NDArray | None,
    transform: Transform,
    *,
    task: Task,
    output_format: OutputFormat,
    layout: Layout,
    num_classes: int | None,
    conf: float,
    iou: float,
    max_det: int,
    classes: tuple[int, ...] | None,
    agnostic_nms: bool,
    mask_threshold: float,
    names: Mapping[int, str],
    active_classes: int | None = None,
) -> Results:
    """Decode one image and return aligned, consistently shaped arrays."""
    extra = 1 if task == "obb" else 0
    proto = None
    if task == "segment":
        if (
            prototypes is None
            or prototypes.ndim != 4
            or prototypes.shape[0] != 1
            or min(prototypes.shape[1:]) < 1
        ):
            raise ValueError("Mask prototypes must have shape (1, channels, height, width).")
        proto = prototypes[0].astype(np.float32, copy=False)
        if not np.isfinite(proto).all():
            raise ValueError("Mask prototypes contain non-finite values.")
        extra = proto.shape[0]
    rows, processed = prediction_rows(output, extra, output_format, layout, num_classes)
    rows = rows[np.isfinite(rows).all(axis=1)]
    if processed:
        scores = rows[:, 4]
        labels = rows[:, 5]
        valid_labels = (labels >= 0) & (labels < 2**31) & (labels == np.floor(labels))
        if num_classes is not None:
            unknown = (labels == np.floor(labels)) & (labels >= num_classes)
            if np.any(unknown & (scores > 0) & (scores <= 1)):
                raise ValueError(
                    "An output class ID exceeds the declared class count. "
                    "Check the ONNX metadata and class names against the training dataset."
                )
            valid_labels &= labels < num_classes
        if active_classes is not None:
            valid_labels &= labels < active_classes
        rows, scores, labels = rows[valid_labels], scores[valid_labels], labels[valid_labels]
        class_ids = labels.astype(np.int64)
        extras = rows[:, 6:]
    else:
        count = rows.shape[1] - 4 - extra
        probabilities = rows[:, 4 : 4 + count]
        if active_classes is not None:
            if not 1 <= active_classes <= count:
                raise ValueError("Active prompt count exceeds raw output class channels.")
            probabilities = probabilities[:, :active_classes]
        if np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("Raw class scores must be probabilities in [0, 1], not logits.")
        class_ids = probabilities.argmax(axis=1).astype(np.int64)
        scores = probabilities.max(axis=1)
        extras = rows[:, 4 + count :]
    keep = (scores > conf) & (scores <= 1)
    if classes is not None:
        keep &= np.isin(class_ids, classes)
    rows, scores, class_ids, extras = rows[keep], scores[keep], class_ids[keep], extras[keep]
    if task == "obb":
        boxes = np.concatenate((rows[:, :4], extras[:, :1]), axis=1)
        valid = (boxes[:, 2:4] > 0).all(axis=1)
    else:
        boxes = rows[:, :4].copy() if processed else xywh_to_xyxy(rows[:, :4])
        valid = (boxes[:, 2:4] > boxes[:, :2]).all(axis=1)
    boxes, scores, class_ids, extras = boxes[valid], scores[valid], class_ids[valid], extras[valid]
    obb = None
    if task == "obb":
        obb = transform.restore_obb(boxes)
        corners = obb_to_corners(obb)
        restored = np.concatenate((corners.min(axis=1), corners.max(axis=1)), axis=1)
        restored = clip_boxes(restored, transform.original_shape)
    else:
        restored = transform.restore_boxes(boxes)
    visible = (restored[:, 2:4] > restored[:, :2]).all(axis=1)
    boxes, scores, class_ids, extras, restored = (
        values[visible] for values in (boxes, scores, class_ids, extras, restored)
    )
    if obb is not None:
        obb = obb[visible]
    if processed:
        indices = np.argsort(-scores, kind="stable")[:max_det]
    else:
        indices = non_max_suppression(
            boxes, scores, class_ids, iou, max_det, agnostic_nms, task == "obb"
        )
    boxes, scores, class_ids, extras = (
        values[indices] for values in (restored, scores, class_ids, extras)
    )
    if obb is not None:
        obb = obb[indices]
    masks = (
        None if proto is None else decode_masks(proto, extras, boxes, transform, mask_threshold)
    )
    return Results(boxes, scores, class_ids, transform.original_shape, masks, obb, names=names)
