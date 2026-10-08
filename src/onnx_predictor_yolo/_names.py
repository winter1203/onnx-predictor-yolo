"""Resolve class labels without inventing or reordering class IDs."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path

import yaml

Names = Mapping[int, str] | Sequence[str] | str | Path


def class_names(value: object) -> dict[int, str]:
    """Validate labels and retain their declared zero-based class IDs."""
    if isinstance(value, Mapping):
        names = {}
        for key, label in value.items():
            if isinstance(key, str) and key.isascii() and key.isdecimal():
                key = int(key)
            if isinstance(key, bool) or not isinstance(key, Integral) or key < 0:
                raise ValueError("Class name keys must be non-negative integer IDs.")
            if int(key) in names:
                raise ValueError("Class name IDs must be unique.")
            names[int(key)] = label
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        names = dict(enumerate(value))
    else:
        raise ValueError("names must contain a nonempty sequence or mapping of class labels.")
    if not names or set(names) != set(range(len(names))):
        raise ValueError("names must be nonempty with contiguous IDs starting at zero.")
    if any(not isinstance(label, str) or not label.strip() for label in names.values()):
        raise ValueError("Every class name must be a nonempty string.")
    return {key: names[key] for key in range(len(names))}


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys instead of silently replacing their values."""

    def construct_mapping(self, node, deep=False):
        if not isinstance(node, yaml.MappingNode):
            raise ValueError("Expected a YAML mapping node.")
        self.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in result
            except TypeError as exc:
                raise ValueError("YAML mapping keys must be scalar values.") from exc
            if duplicate:
                raise ValueError(f"Duplicate YAML key: {key!r}.")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def configured_names(value: Names) -> dict[int, str]:
    """Load explicit labels or the names field of a training dataset YAML."""
    if not isinstance(value, (str, Path)):
        return class_names(value)
    path = Path(value)
    try:
        data = yaml.load(path.read_text(encoding="utf-8-sig"), Loader=_UniqueKeyLoader)
    except (OSError, UnicodeError, yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"Cannot read class names from YAML {str(path)!r}: {exc}") from exc
    if not isinstance(data, Mapping) or "names" not in data:
        raise ValueError(f"Dataset YAML {str(path)!r} must contain a 'names' field.")
    names = class_names(data["names"])
    if "nc" in data:
        count = data["nc"]
        if isinstance(count, bool) or not isinstance(count, Integral) or count != len(names):
            raise ValueError("Dataset YAML 'nc' must equal the number of class names.")
    return names


def resolve_names(
    metadata: Mapping[str, str], explicit: Names | None, *, prompted: bool
) -> dict[int, str]:
    """Read metadata first, then use an explicit configuration when necessary."""
    model_names = None
    metadata_error = None
    try:
        if "names" not in metadata:
            raise ValueError("The ONNX model has no 'names' metadata.")
        serialized = metadata["names"]
        if not isinstance(serialized, str):
            raise ValueError("Class name metadata must be a serialized list or mapping.")
        expression = ast.parse(serialized.strip(), mode="eval").body
        value = ast.literal_eval(expression)
        if isinstance(expression, ast.Dict) and len(expression.keys) != len(value):
            raise ValueError("Class name IDs in metadata must be unique.")
        model_names = class_names(value)
    except (ValueError, SyntaxError, TypeError) as exc:
        metadata_error = exc
    if explicit is not None:
        supplied = configured_names(explicit)
        if model_names is not None and not prompted and supplied != model_names:
            raise ValueError(
                "Configured names conflict with ONNX metadata. Use the training class order "
                "and correct the export metadata instead of relabeling model outputs."
            )
        return supplied
    if model_names is None:
        raise ValueError(
            "Cannot resolve class names from ONNX metadata. Supply names in training class "
            "order as a list, an ID mapping, or a dataset YAML path. "
            f"Metadata error: {metadata_error}"
        ) from metadata_error
    return model_names
