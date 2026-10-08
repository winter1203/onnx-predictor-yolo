"""Reusable ONNX text and visual encoders for YOLOE prompt embeddings."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

from ._geometry import clip_boxes, padding_color, preprocess
from ._runtime import checked_tensor, fixed_batch, load_session, tensor_dtype
from ._runtime import image_input as select_image_input
from .predictor import _input_size, _positive_int

if TYPE_CHECKING:
    from onnxruntime import InferenceSession, SessionOptions


class TextPromptEncoder:
    """Encode text or token IDs with an externally supplied ONNX text encoder.

    The tokenizer must match the exported encoder and return an integer NumPy
    array of shape (number_of_prompts, context_length). It is an application
    callback; no tokenizer, training framework, or weights are downloaded.
    Token arrays may be supplied directly without a tokenizer.
    """

    def __init__(
        self,
        model: str | Path | InferenceSession,
        *,
        tokenizer: Callable[[Sequence[str]], NDArray] | None = None,
        normalize: bool = False,
        providers: Sequence[str] | None = None,
        session_options: SessionOptions | None = None,
    ) -> None:
        self.session = load_session(model, providers, session_options)
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(inputs[0].shape) != 2:
            raise ValueError(
                "Text encoders require one token input with shape (Q, context_length)."
            )
        if inputs[0].type not in ("tensor(int32)", "tensor(int64)"):
            raise ValueError("Text encoder token inputs must be int32 or int64.")
        if len(outputs) != 1 or len(outputs[0].shape) != 2:
            raise ValueError("Text encoders require one embedding output with shape (Q, D).")
        if tokenizer is not None and not callable(tokenizer):
            raise TypeError("tokenizer must be callable.")
        if not isinstance(normalize, bool):
            raise TypeError("normalize must be a bool.")
        self._input = inputs[0]
        self._output_name = outputs[0].name
        self._batch_size = fixed_batch(inputs[0])
        self.tokenizer = tokenizer
        self.normalize = normalize

    def encode(self, prompts: Sequence[str] | NDArray) -> NDArray[np.float32]:
        """Return embeddings shaped (1, Q, D), preserving prompt order."""
        if isinstance(prompts, np.ndarray):
            tokens = prompts
        else:
            if (
                isinstance(prompts, (str, bytes))
                or not isinstance(prompts, Sequence)
                or not prompts
                or any(not isinstance(text, str) or not text for text in prompts)
            ):
                raise ValueError(
                    "prompts must be a nonempty sequence of text labels or token array."
                )
            if self.tokenizer is None:
                raise ValueError("Supply a matching tokenizer or pass token IDs directly.")
            tokens = np.asarray(self.tokenizer(prompts))
            if tokens.ndim != 2 or tokens.shape[0] != len(prompts):
                raise ValueError("The tokenizer must return exactly one token row per prompt.")
        if tokens.ndim != 2 or min(tokens.shape) < 1:
            raise ValueError("Token IDs must have nonempty shape (Q, context_length).")
        chunk_size = self._batch_size or len(tokens)
        embeddings = []
        for start in range(0, len(tokens), chunk_size):
            chunk = tokens[start : start + chunk_size]
            count = len(chunk)
            if count < chunk_size:
                chunk = np.concatenate((chunk, np.repeat(chunk[-1:], chunk_size - count, axis=0)))
            output = self.session.run(
                [self._output_name], {self._input.name: checked_tensor(chunk, self._input)}
            )[0]
            if output.ndim != 2 or output.shape[0] != len(chunk) or output.shape[1] < 1:
                raise ValueError("Text encoder output must have shape (submitted_prompts, D).")
            embeddings.append(output[:count].astype(np.float32))
        result = np.concatenate(embeddings, axis=0)
        if not np.isfinite(result).all():
            raise ValueError("Text embeddings contain non-finite values.")
        if self.normalize:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            if not np.isfinite(norms).all() or np.any(norms == 0):
                raise ValueError("Cannot normalize zero or non-finite embedding norms.")
            result /= norms
        return result[None]

    __call__ = encode


class VisualPromptEncoder:
    """Extract reusable visual embeddings from reference-image boxes or masks.

    The ONNX graph must expose an RGB NCHW image input, a floating-point
    (B, Q, mask_height, mask_width) input, and a (B, Q, D) embedding output.
    Multiple reference regions with the same class ID are combined by union.
    The returned class order is 0, 1, ..., Q-1; IDs must be contiguous.
    """

    def __init__(
        self,
        model: str | Path | InferenceSession,
        *,
        imgsz: int | tuple[int, int] | None = None,
        pad_value: int | tuple[int, int, int] = 114,
        mask_stride: int = 8,
        image_input: str | None = None,
        providers: Sequence[str] | None = None,
        session_options: SessionOptions | None = None,
    ) -> None:
        self.session = load_session(model, providers, session_options)
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        self._image = select_image_input(inputs, image_input)
        masks = [node for node in inputs if node.name != self._image.name]
        if len(masks) != 1 or len(masks[0].shape) != 4:
            raise ValueError(
                "Visual encoders require one image input and one BQHW prompt mask input."
            )
        self._mask = masks[0]
        for node in (self._image, self._mask):
            if node.type not in ("tensor(float)", "tensor(float16)"):
                raise ValueError("Visual encoder inputs must be float32 or float16.")
        if len(outputs) != 1 or len(outputs[0].shape) != 3:
            raise ValueError("Visual encoders require one embedding output with shape (B, Q, D).")
        self._output_name = outputs[0].name
        image_batch, mask_batch = fixed_batch(self._image), fixed_batch(self._mask)
        if image_batch is not None and mask_batch is not None and image_batch != mask_batch:
            raise ValueError("Visual encoder image and mask batches must match.")
        self._batch_size = image_batch or mask_batch or 1
        self.imgsz = _input_size(
            self._image.shape, imgsz, self.session.get_modelmeta().custom_metadata_map
        )
        self.pad_value = padding_color(pad_value)
        mask_stride = _positive_int(mask_stride, "mask_stride")
        dimensions = []
        for declared, size in zip(self._mask.shape[2:], self.imgsz, strict=True):
            if isinstance(declared, Integral):
                dimensions.append(_positive_int(declared, "mask dimension"))
            else:
                if size % mask_stride:
                    raise ValueError(
                        "Dynamic mask dimensions require imgsz divisible by mask_stride."
                    )
                dimensions.append(_positive_int(size // mask_stride, "mask dimension"))
        self.mask_shape = (dimensions[0], dimensions[1])
        capacity = self._mask.shape[1]
        self._capacity = (
            _positive_int(capacity, "prompt capacity") if isinstance(capacity, Integral) else None
        )

    def encode(
        self,
        image: NDArray[np.uint8],
        *,
        boxes: NDArray | Sequence[Sequence[float]] | None = None,
        masks: NDArray | None = None,
        class_ids: Sequence[int] | NDArray | None = None,
    ) -> NDArray[np.float32]:
        """Return (1, Q, D) embeddings for one reference image, with no padded classes.

        Supply either xyxy boxes in original pixels or binary instance masks
        shaped (number_of_regions, original_height, original_width). Without
        class IDs each region defines a separate class.
        """
        if (boxes is None) == (masks is None):
            raise ValueError("Supply exactly one of boxes or masks.")
        tensor, transform = preprocess(
            image, self.imgsz, tensor_dtype(self._image).type, "letterbox", self.pad_value
        )
        if boxes is not None:
            regions = np.asarray(boxes, dtype=np.float32).copy()
            if regions.ndim != 2 or regions.shape[1] != 4 or not np.isfinite(regions).all():
                raise ValueError("boxes must be a finite (N, 4) xyxy array.")
            regions = clip_boxes(regions, transform.original_shape)
            if np.any(regions[:, 2:] <= regions[:, :2]):
                raise ValueError("Reference boxes must have positive area inside the image.")
        else:
            regions = np.asarray(masks)
            if (
                regions.ndim != 3
                or regions.shape[1:] != transform.original_shape
                or regions.dtype.kind not in "buif"
                or not np.isfinite(regions).all()
            ):
                raise ValueError(
                    "masks must be finite binary arrays shaped (N, image_height, image_width)."
                )
            if not np.isin(regions, (0, 1, 255)).all():
                raise ValueError("Reference masks must contain only 0, 1, or 255.")
        count = len(regions)
        if count == 0:
            raise ValueError("At least one reference region is required.")
        labels = np.arange(count) if class_ids is None else np.asarray(class_ids)
        if labels.shape != (count,) or labels.dtype.kind not in "iu" or np.any(labels < 0):
            raise ValueError(
                "class_ids must contain one non-negative integer per reference region."
            )
        unique = np.unique(labels)
        if not np.array_equal(unique, np.arange(len(unique))):
            raise ValueError("Reference class IDs must be contiguous and start at zero.")
        active = len(unique)
        capacity = self._capacity or active
        if active > capacity:
            raise ValueError(f"The visual encoder supports at most {capacity} classes.")
        mask_h, mask_w = self.mask_shape
        input_h, input_w = self.imgsz
        left, top = transform.padding
        resized_h, resized_w = transform.resized_shape
        prompt_masks = np.zeros((1, capacity, mask_h, mask_w), dtype=np.float32)
        grid_x, grid_y = np.arange(mask_w)[None, :], np.arange(mask_h)[:, None]
        for region, label in zip(regions, labels, strict=True):
            if boxes is not None:
                x1, y1, x2, y2 = region
                x1, x2 = (np.array([x1, x2]) * transform.scale[0] + left) * mask_w / input_w
                y1, y2 = (np.array([y1, y2]) * transform.scale[1] + top) * mask_h / input_h
                mask = (grid_x >= x1) & (grid_x < x2) & (grid_y >= y1) & (grid_y < y2)
            else:
                resized = cv2.resize(
                    (region != 0).astype(np.uint8),
                    (resized_w, resized_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                canvas = np.zeros(self.imgsz, dtype=np.uint8)
                canvas[top : top + resized_h, left : left + resized_w] = resized
                mask = cv2.resize(canvas, (mask_w, mask_h), interpolation=cv2.INTER_NEAREST) != 0
            prompt_masks[0, int(label)] = np.maximum(prompt_masks[0, int(label)], mask)
        if not prompt_masks[0, :active].any(axis=(1, 2)).all():
            raise ValueError("Every class needs a nonempty region at the encoder mask resolution.")
        feeds = {
            self._image.name: checked_tensor(
                np.repeat(tensor, self._batch_size, axis=0), self._image
            ),
            self._mask.name: checked_tensor(
                np.repeat(prompt_masks, self._batch_size, axis=0), self._mask
            ),
        }
        output = self.session.run([self._output_name], feeds)[0]
        if (
            output.ndim != 3
            or output.shape[:2] != (self._batch_size, capacity)
            or output.shape[2] < 1
        ):
            raise ValueError(
                "Visual encoder output must have shape (submitted_batch, capacity, D)."
            )
        result = output[:1, :active].astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Visual embeddings contain non-finite values.")
        return result

    __call__ = encode
