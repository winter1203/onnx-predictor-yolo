"""Structured inference results in original image coordinates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from ._geometry import obb_to_corners
from ._names import class_names


@dataclass(frozen=True)
class Results:
    """Predictions for one image, ordered by descending confidence.

    Boxes are float32 ``(N, 4)`` arrays in clipped ``xyxy`` pixel coordinates.
    Scores are float32 ``(N,)`` arrays; class IDs are int64 ``(N,)`` arrays.
    ``names`` is a read-only vocabulary; ``class_names`` follows class ID order.
    Segmentation masks are boolean ``(N, H, W)`` arrays, otherwise ``None``.
    Oriented boxes are float32 ``(N, 5)`` arrays containing center x, center y,
    width, height, and rotation in radians, otherwise ``None``. Their geometry
    is not clipped; ``boxes`` contains their clipped axis-aligned envelopes.
    Empty predictions retain these shapes. Arrays themselves remain mutable.
    """

    boxes: NDArray[np.float32]
    scores: NDArray[np.float32]
    class_ids: NDArray[np.int64]
    image_shape: tuple[int, int]
    masks: NDArray[np.bool_] | None = None
    obb: NDArray[np.float32] | None = None
    names: Mapping[int, str] = field(kw_only=True)

    def __post_init__(self) -> None:
        """Capture validated labels so later prediction calls cannot change them."""
        names = class_names(self.names)
        if self.class_ids.dtype.kind not in "iu" or any(
            int(class_id) not in names for class_id in self.class_ids
        ):
            raise ValueError("Every result class ID must have a corresponding class name.")
        object.__setattr__(self, "names", MappingProxyType(names))

    @property
    def class_names(self) -> tuple[str, ...]:
        """Return labels aligned with class_ids, without numeric-name fallbacks."""
        return tuple(self.names[int(class_id)] for class_id in self.class_ids)

    def __len__(self) -> int:
        """Return the number of predictions."""
        return len(self.scores)

    @property
    def polygons(self) -> NDArray[np.float32] | None:
        """Return oriented box corners as ``(N, 4, 2)``, or ``None``."""
        return None if self.obb is None else obb_to_corners(self.obb)
