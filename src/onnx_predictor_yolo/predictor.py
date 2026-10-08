"""A reusable ONNX Runtime session with one public prediction interface."""

from __future__ import annotations

import ast
import asyncio
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from functools import partial
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from numpy.typing import NDArray

from ._geometry import padding_color, preprocess
from ._names import Names, configured_names, resolve_names
from ._postprocess import Layout, OutputFormat, Task, postprocess
from ._runtime import checked_tensor, fixed_batch, load_session
from ._runtime import image_input as select_image_input
from .results import Results

if TYPE_CHECKING:
    from onnxruntime import InferenceSession, SessionOptions


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _metadata_value(metadata: Mapping[str, str], key: str) -> object:
    value = metadata.get(key)
    if value is None:
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Invalid {key!r} model metadata; supply an explicit override.") from exc


def _input_size(
    shape: Sequence, imgsz: int | tuple[int, int] | None, metadata: Mapping[str, str]
) -> tuple[int, int]:
    if len(shape) != 4:
        raise ValueError("Only NCHW image inputs with rank 4 are supported.")
    if isinstance(shape[1], Integral) and shape[1] != 3:
        raise ValueError("The model must accept three RGB channels in NCHW layout.")
    spatial = shape[2:]
    if imgsz is None and all(isinstance(dim, Integral) and dim > 0 for dim in spatial):
        return int(spatial[0]), int(spatial[1])
    if imgsz is None:
        imgsz = _metadata_value(metadata, "imgsz")
    if imgsz is None:
        raise ValueError("Dynamic input dimensions require imgsz=(height, width).")
    if isinstance(imgsz, Integral):
        size = (_positive_int(imgsz, "imgsz"),) * 2
    elif isinstance(imgsz, (tuple, list)) and len(imgsz) == 2:
        size = tuple(_positive_int(dim, "imgsz dimension") for dim in imgsz)
    else:
        raise ValueError("imgsz must be a positive integer or a (height, width) pair.")
    for declared, requested in zip(spatial, size, strict=True):
        if isinstance(declared, Integral) and declared != requested:
            raise ValueError(f"imgsz {size} conflicts with the model input shape {shape}.")
    return size[0], size[1]


class YoloPredictor:
    """Run YOLO/YOLOE detection, segmentation, or OBB on images and batches.

    A model is a local ONNX path or an existing ONNX Runtime InferenceSession.
    CPU execution and letterbox preprocessing are the defaults. Task metadata
    is used when available; metadata-free OBB models need ``task='obb'``.
    ``pad_value`` accepts a gray level or a BGR tuple, defaulting to gray 114.
    Class names are required: read ONNX metadata or supply ``names`` as a
    training-order sequence, ID mapping, or dataset YAML path. Fixed-vocabulary
    metadata and explicit names must agree. Prompt models may supply labels
    for their current embeddings instead of the exported vocabulary.
    Raw predictions contain center/size boxes and class probabilities, with
    optional mask coefficients or angles. Processed predictions contain
    boxes, score, class ID, and task extras. ``output_format='end2end'`` also
    accepts compatible exports with embedded NMS. See the README for the
    exact tensor contracts and the explicit overrides for ambiguous exports.
    """

    def __init__(
        self,
        model: str | Path | InferenceSession,
        *,
        task: Task | None = None,
        output_format: OutputFormat = "auto",
        layout: Layout = "channels_first",
        imgsz: int | tuple[int, int] | None = None,
        resize: Literal["letterbox", "stretch"] = "letterbox",
        pad_value: int | tuple[int, int, int] = 114,
        names: Names | None = None,
        num_classes: int | None = None,
        providers: Sequence[str] | None = None,
        session_options: SessionOptions | None = None,
        image_input: str | None = None,
    ) -> None:
        if task is not None and task not in ("detect", "segment", "obb"):
            raise ValueError("task must be 'detect', 'segment', or 'obb'.")
        if output_format not in ("auto", "raw", "end2end"):
            raise ValueError("output_format must be 'auto', 'raw', or 'end2end'.")
        if layout not in ("channels_first", "channels_last"):
            raise ValueError("layout must be 'channels_first' or 'channels_last'.")
        if resize not in ("letterbox", "stretch"):
            raise ValueError("resize must be 'letterbox' or 'stretch'.")
        self.pad_value = padding_color(pad_value)
        self.session = load_session(model, providers, session_options)
        inputs = self.session.get_inputs()
        model_input = select_image_input(inputs, image_input)
        auxiliary = [item for item in inputs if item.name != model_input.name]
        if len(auxiliary) > 1 or (auxiliary and len(auxiliary[0].shape) != 3):
            raise ValueError(
                "Expected an image input and optionally a YOLOE embedding input (B, Q, D)."
            )
        self._image_input = model_input
        self._prompt_input = auxiliary[0] if auxiliary else None
        self.prompt_input_name = auxiliary[0].name if auxiliary else None
        self.batch_size = fixed_batch(model_input)
        if self._prompt_input is not None:
            if self._prompt_input.type not in ("tensor(float)", "tensor(float16)"):
                raise ValueError("YOLOE embeddings must be float32 or float16.")
            prompt_batch = fixed_batch(self._prompt_input)
            if prompt_batch not in (None, 1):
                if self.batch_size not in (None, prompt_batch):
                    raise ValueError("Image and prompt batch dimensions are incompatible.")
                self.batch_size = prompt_batch
        dtypes = {"tensor(float)": np.float32, "tensor(float16)": np.float16}
        if model_input.type not in dtypes:
            raise ValueError("The model input must be float32 or float16.")
        self._dtype = dtypes[model_input.type]
        self._input_name = model_input.name
        self.metadata = dict(self.session.get_modelmeta().custom_metadata_map)
        self.imgsz = _input_size(model_input.shape, imgsz, self.metadata)
        self._names = MappingProxyType(
            resolve_names(self.metadata, names, prompted=bool(auxiliary))
        )
        if num_classes is not None:
            num_classes = _positive_int(num_classes, "num_classes")
        if not auxiliary and num_classes is not None and len(self.names) != num_classes:
            raise ValueError("num_classes must match the number of class names.")
        if auxiliary:
            capacity = auxiliary[0].shape[1]
            capacity = int(capacity) if isinstance(capacity, Integral) else None
            if capacity is not None and capacity < 1:
                raise ValueError("Prompt capacity must be positive.")
            if num_classes is not None and capacity is not None and num_classes != capacity:
                raise ValueError("num_classes must match the exported prompt capacity.")
            self.num_classes = num_classes if num_classes is not None else capacity
            if self.num_classes is not None and len(self.names) > self.num_classes:
                raise ValueError("The number of class names exceeds the exported prompt capacity.")
        else:
            self.num_classes = len(self.names)
        outputs = self.session.get_outputs()
        predictions = [output for output in outputs if len(output.shape) == 3]
        prototypes = [output for output in outputs if len(output.shape) == 4]
        resolved_task = task or self.metadata.get("task") or ("segment" if prototypes else "detect")
        if resolved_task not in ("detect", "segment", "obb"):
            raise ValueError(f"Unsupported model task: {resolved_task!r}.")
        if resolved_task == "obb" and resize != "letterbox":
            raise ValueError("OBB requires resize='letterbox' to preserve rotated rectangles.")
        if len(predictions) != 1 or len(outputs) != (2 if resolved_task == "segment" else 1):
            raise ValueError(
                "Expected one prediction output and, for segmentation, one prototype output."
            )
        if resolved_task == "segment" and len(prototypes) != 1:
            raise ValueError("Segmentation requires an NCHW mask prototype output.")
        self._prediction_name = predictions[0].name
        self._prototype_name = prototypes[0].name if prototypes else None
        self.task = cast(Task, resolved_task)
        self.output_format = output_format
        self.layout = layout
        self.resize = resize

    @property
    def names(self) -> Mapping[int, str]:
        """Read-only class labels indexed by the model's original class IDs."""
        return self._names

    def predict(
        self,
        image: NDArray[np.uint8] | Sequence[NDArray[np.uint8]],
        *,
        batch_size: int | None = None,
        embeddings: NDArray | None = None,
        prompt_names: Names | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
        classes: Sequence[int] | None = None,
        agnostic_nms: bool = False,
        mask_threshold: float = 0.5,
    ) -> Results | list[Results]:
        """Predict an image or batch of BGR uint8 images, preserving input order.

        An HWC array returns Results; a sequence or BHWC array returns a list.
        ``batch_size`` limits real images per session call. Fixed-size models
        repeat the last sample to fill short batches and discard padded results.
        YOLOE ``embeddings`` accepts shared (Q, D)/(1, Q, D) prompts or per-image
        (B, Q, D) prompts. Unused fixed-capacity prompt slots are zero-padded
        and excluded from class selection. ``prompt_names`` overrides labels
        for this call only and must follow embedding order; it is YOLOE-only.

        Confidence uses a strict ``score > conf`` comparison. Raw predictions
        use class-aware NMS unless ``agnostic_nms=True``. Processed exports
        skip NMS, so ``iou`` and ``agnostic_nms`` have no effect on them.
        ``classes`` filters integer class IDs before suppression and limiting.
        ``mask_threshold`` is a probability threshold applied to interpolated
        mask logits. All returned coordinates refer to the original image.
        """
        for name, value in (("conf", conf), ("iou", iou)):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and within [0, 1].")
        if not np.isfinite(mask_threshold) or not 0 < mask_threshold < 1:
            raise ValueError("mask_threshold must be finite and strictly between 0 and 1.")
        max_det = _positive_int(max_det, "max_det")
        if batch_size is not None:
            batch_size = _positive_int(batch_size, "batch_size")
        if not isinstance(agnostic_nms, bool):
            raise TypeError("agnostic_nms must be a bool.")
        class_filter = None
        if classes is not None:
            if any(
                isinstance(value, bool) or not isinstance(value, Integral) or value < 0
                for value in classes
            ):
                raise ValueError("classes must contain non-negative integer IDs.")
            class_filter = tuple(int(value) for value in classes)
            if self.num_classes is not None and any(
                value >= self.num_classes for value in class_filter
            ):
                raise ValueError("A requested class ID exceeds the model class count.")
        single = isinstance(image, np.ndarray) and image.ndim == 3
        if isinstance(image, np.ndarray) and image.ndim not in (3, 4):
            raise ValueError("image must be HWC or BHWC; use a sequence for varying image sizes.")
        if not isinstance(image, (np.ndarray, Sequence)) or isinstance(image, (str, bytes)):
            raise TypeError("image must be a NumPy image or a sequence of NumPy images.")
        images = [image] if single else list(image)
        if not images:
            return []
        if prompt_names is not None and self._prompt_input is None:
            raise ValueError("prompt_names requires a YOLOE embedding input.")
        names = self.names if prompt_names is None else configured_names(prompt_names)
        prompts, active_classes, model_classes = self._prepare_prompts(
            embeddings, len(images), names
        )
        if (
            class_filter is not None
            and active_classes is not None
            and any(value >= active_classes for value in class_filter)
        ):
            raise ValueError("A requested class ID exceeds the active prompt count.")
        chunk_size = min(batch_size or len(images), self.batch_size or len(images))
        shared_input = self._prompt_input is not None and fixed_batch(self._prompt_input) == 1
        if shared_input and prompts is not None and prompts.shape[0] != 1:
            chunk_size = 1
        output_names = [self._prediction_name]
        if self._prototype_name is not None:
            output_names.append(self._prototype_name)
        results = []
        for start in range(0, len(images), chunk_size):
            chunk = images[start : start + chunk_size]
            prepared = [
                preprocess(item, self.imgsz, self._dtype, self.resize, self.pad_value)
                for item in chunk
            ]
            run_size = self.batch_size or len(chunk)
            tensors = [item[0] for item in prepared]
            tensors.extend([tensors[-1]] * (run_size - len(chunk)))
            feeds = {self._input_name: checked_tensor(np.concatenate(tensors), self._image_input)}
            if prompts is not None:
                selected = prompts if prompts.shape[0] == 1 else prompts[start : start + len(chunk)]
                if not shared_input:
                    if selected.shape[0] == 1:
                        selected = np.repeat(selected, run_size, axis=0)
                    elif len(chunk) < run_size:
                        selected = np.concatenate(
                            (selected, np.repeat(selected[-1:], run_size - len(chunk), axis=0))
                        )
                feeds[self.prompt_input_name] = checked_tensor(selected, self._prompt_input)
            outputs = self.session.run(output_names, feeds)
            if any(output.ndim < 1 or output.shape[0] != run_size for output in outputs):
                raise ValueError("Output batch dimensions do not match the submitted image batch.")
            for index, (_, transform) in enumerate(prepared):
                results.append(
                    postprocess(
                        outputs[0][index : index + 1],
                        outputs[1][index : index + 1] if self._prototype_name is not None else None,
                        transform,
                        task=self.task,
                        output_format=self.output_format,
                        layout=self.layout,
                        num_classes=model_classes,
                        conf=conf,
                        iou=iou,
                        max_det=max_det,
                        classes=class_filter,
                        agnostic_nms=agnostic_nms,
                        mask_threshold=mask_threshold,
                        active_classes=active_classes,
                        names=names,
                    )
                )
        return results[0] if single else results

    async def predict_async(
        self,
        image: NDArray[np.uint8] | Sequence[NDArray[np.uint8]],
        *,
        batch_size: int | None = None,
        embeddings: NDArray | None = None,
        prompt_names: Names | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
        classes: Sequence[int] | None = None,
        agnostic_nms: bool = False,
        mask_threshold: float = 0.5,
        executor: ThreadPoolExecutor | None = None,
    ) -> Results | list[Results]:
        """Run the complete prediction pipeline in a worker thread.

        Prediction options and results match ``predict``. Pass a caller-owned
        thread pool to limit concurrent calls, or use the event loop's default
        executor. Context variables propagate to the worker. The predictor
        never shuts down a supplied executor.

        Do not mutate inputs, embeddings, or predictor/session configuration
        while work is in flight. Cancellation stops awaiting the result but
        cannot interrupt an already running worker or ONNX Runtime call.
        Concurrent session calls require support from the execution provider;
        use a single-worker pool for providers that require serialization.
        """
        if executor is not None and not isinstance(executor, ThreadPoolExecutor):
            raise TypeError("executor must be a ThreadPoolExecutor or None.")
        call = partial(
            self.predict,
            image,
            batch_size=batch_size,
            embeddings=embeddings,
            prompt_names=prompt_names,
            conf=conf,
            iou=iou,
            max_det=max_det,
            classes=classes,
            agnostic_nms=agnostic_nms,
            mask_threshold=mask_threshold,
        )
        return await asyncio.get_running_loop().run_in_executor(executor, copy_context().run, call)

    def _prepare_prompts(
        self, embeddings: NDArray | None, count: int, names: Mapping[int, str]
    ) -> tuple[NDArray | None, int | None, int | None]:
        """Validate prompt batches and pad only unused class slots."""
        if self._prompt_input is None:
            if embeddings is not None:
                raise ValueError("This export has no embedding input; prompts may be baked in.")
            return None, None, self.num_classes
        if embeddings is None:
            raise ValueError(f"embeddings are required for input {self.prompt_input_name!r}.")
        prompts = np.asarray(embeddings)
        if prompts.ndim == 2:
            prompts = prompts[None]
        if prompts.ndim != 3 or prompts.shape[0] not in (1, count) or min(prompts.shape[1:]) < 1:
            raise ValueError("embeddings must have shape (Q, D), (1, Q, D), or (B, Q, D).")
        if prompts.dtype.kind != "f" or not np.isfinite(prompts).all():
            raise ValueError("embeddings must contain finite floating-point values.")
        active = prompts.shape[1]
        capacity = self.num_classes or active
        if active > capacity:
            raise ValueError(f"Received {active} prompts, but the export supports {capacity}.")
        if len(names) != active:
            raise ValueError("names must match the number and order of active prompts.")
        padded = np.zeros((prompts.shape[0], capacity, prompts.shape[2]), dtype=prompts.dtype)
        padded[:, :active] = prompts
        return padded, active, capacity

    __call__ = predict
