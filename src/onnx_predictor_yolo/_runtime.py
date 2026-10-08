"""Shared ONNX Runtime loading and tensor contract validation."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from onnxruntime import InferenceSession, NodeArg, SessionOptions


def load_session(
    model: str | Path | InferenceSession,
    providers: Sequence[str] | None,
    options: SessionOptions | None,
) -> InferenceSession:
    """Reuse a session or create one without concealing provider failures."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError("Install onnx-predictor-yolo[cpu] or onnx-predictor-yolo[gpu].") from exc
    if isinstance(model, ort.InferenceSession):
        if providers is not None or options is not None:
            raise ValueError("Configure providers and options on the supplied session itself.")
        return model
    if not isinstance(model, (str, Path)):
        raise TypeError("model must be an ONNX path or an ONNX Runtime InferenceSession.")
    if providers is not None and (isinstance(providers, str) or not providers):
        raise ValueError("providers must be a nonempty sequence of provider names.")
    selected = list(providers) if providers is not None else ["CPUExecutionProvider"]
    if any(not isinstance(provider, str) for provider in selected):
        raise TypeError("Provider names must be strings.")
    missing = set(selected) - set(ort.get_available_providers())
    if missing:
        raise ValueError(f"Requested ONNX Runtime providers are unavailable: {sorted(missing)}")
    session = ort.InferenceSession(str(model), sess_options=options, providers=selected)
    if session.get_providers()[0] != selected[0]:
        raise RuntimeError(f"Requested provider {selected[0]} did not initialize.")
    return session


def image_input(inputs: Sequence[NodeArg], name: str | None = None) -> NodeArg:
    """Select a single image input, allowing an explicit name for ambiguous graphs."""
    if name is not None:
        matches = [item for item in inputs if item.name == name]
    else:
        matches = [
            item
            for item in inputs
            if len(item.shape) == 4
            and item.name in ("images", "image", "x")
        ]
        if not matches:
            matches = [
                item
                for item in inputs
                if len(item.shape) == 4
                and (not isinstance(item.shape[1], Integral) or item.shape[1] == 3)
            ]
    if len(matches) != 1:
        raise ValueError("Cannot identify the image input; set image_input explicitly.")
    if len(matches[0].shape) != 4:
        raise ValueError("The selected image input must have rank four in NCHW layout.")
    return matches[0]


def tensor_dtype(node: NodeArg) -> np.dtype:
    """Return the supported NumPy dtype declared by an input node."""
    types = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
    }
    if node.type not in types:
        raise ValueError(f"Unsupported input type {node.type!r} for {node.name!r}.")
    return np.dtype(types[node.type])


def checked_tensor(value: NDArray, node: NodeArg) -> NDArray:
    """Cast input data, enforcing rank, fixed dimensions, and numeric validity."""
    value = np.asarray(value)
    if value.ndim != len(node.shape) or any(
        isinstance(size, Integral) and size != actual
        for size, actual in zip(node.shape, value.shape, strict=False)
    ):
        raise ValueError(f"Input {node.name!r} expects {node.shape}, received {value.shape}.")
    dtype = tensor_dtype(node)
    if dtype.kind in "iu":
        if value.dtype.kind not in "iu":
            raise TypeError(f"Input {node.name!r} requires integer tokens.")
        bounds = np.iinfo(dtype)
        if np.any(value < bounds.min) or np.any(value > bounds.max):
            raise ValueError(f"Input {node.name!r} exceeds the declared integer range.")
    elif value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError(f"Input {node.name!r} must contain finite numeric values.")
    with np.errstate(over="ignore"):
        result = np.ascontiguousarray(value, dtype=dtype)
    if dtype.kind == "f" and not np.isfinite(result).all():
        raise ValueError(f"Input {node.name!r} overflows its declared dtype.")
    return result


def fixed_batch(node: NodeArg) -> int | None:
    """Return a positive fixed batch dimension, or None for a dynamic one."""
    batch = node.shape[0]
    if isinstance(batch, Integral):
        if batch < 1:
            raise ValueError("Fixed batch dimensions must be positive.")
        return int(batch)
    return None
