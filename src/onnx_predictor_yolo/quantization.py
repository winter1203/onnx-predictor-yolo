"""Offline dynamic quantization with metadata and CPU compatibility checks."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from typing import Literal


def _string_sequence(values: Sequence[str], name: str) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of strings.")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{name} must contain nonempty strings.")
    return list(dict.fromkeys(values))


def quantize_dynamic(
    model: str | Path,
    output: str | Path,
    *,
    op_types: Sequence[str] = ("Conv", "MatMul"),
    weight_type: Literal["uint8", "int8"] = "uint8",
    per_channel: bool = False,
    reduce_range: bool = False,
    nodes_to_exclude: Sequence[str] = (),
) -> Path:
    """Create a CPU-oriented, dynamically quantized ONNX file without calibration.

    Constant FP32 weights of selected Conv/MatMul nodes become 8-bit tensors;
    activations are quantized at inference time. Other operators remain in
    floating point. Dynamic prompt inputs and input/output shapes are retained.
    Conv requires uint8 per-tensor weights for compatibility with the minimum
    supported runtime. MatMul-only conversion also accepts int8/per-channel.

    Original metadata, including class names, is preserved verbatim. Missing
    class names are never invented; YoloPredictor still requires names. Models
    with no eligible weights or no converted integer operators raise errors.

    The source is never overwritten. The output must not exist, and its parent
    directory must exist. Conversion uses temporary files, checks ONNX validity
    and CPU session initialization, then writes one self-contained ONNX file.
    This does not evaluate accuracy or benchmark inference. Output models must
    fit the ONNX single-file size limit; external-data output is not supported.
    """
    operators = _string_sequence(op_types, "op_types")
    excluded = _string_sequence(nodes_to_exclude, "nodes_to_exclude")
    if not operators or set(operators) - {"Conv", "MatMul"}:
        raise ValueError("op_types must select Conv, MatMul, or both.")
    if weight_type not in ("uint8", "int8"):
        raise ValueError("weight_type must be 'uint8' or 'int8'.")
    if not isinstance(per_channel, bool) or not isinstance(reduce_range, bool):
        raise TypeError("per_channel and reduce_range must be bool values.")
    if "Conv" in operators and (weight_type != "uint8" or per_channel):
        raise ValueError(
            "Dynamic Conv requires weight_type='uint8' and per_channel=False. "
            "Use op_types=('MatMul',) for signed or per-channel weights."
        )
    source = Path(model).expanduser().resolve(strict=True)
    destination = Path(output).expanduser().absolute()
    if source == destination.resolve():
        raise ValueError("The quantized output must differ from the source model.")
    if not source.is_file():
        raise ValueError("model must refer to an ONNX file.")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Output already exists: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {destination.parent}")
    try:
        import onnx
        import onnxruntime as ort
        from onnxruntime.quantization import QuantType
        from onnxruntime.quantization import quantize_dynamic as ort_quantize_dynamic
    except ImportError as exc:
        raise ImportError(
            "Install onnx-predictor-yolo[cpu,quantization], or add the quantization "
            "extra to an environment with a compatible ONNX Runtime already installed."
        ) from exc
    graph = onnx.load_model(str(source), load_external_data=True)
    onnx.external_data_helper.convert_model_from_external_data(graph)
    onnx.checker.check_model(graph)
    weights = {item.name: item for item in graph.graph.initializer}
    unknown = set(excluded) - {node.name for node in graph.graph.node}
    if unknown:
        raise ValueError(f"Unknown excluded node names: {sorted(unknown)}")
    candidates = [
        node for node in graph.graph.node
        if node.domain in ("", "ai.onnx") and node.op_type in operators
        and node.name not in excluded and len(node.input) > 1
        and node.input[1] in weights
        and weights[node.input[1]].data_type == onnx.TensorProto.FLOAT
    ]
    if not candidates:
        raise ValueError(
            "No eligible constant FP32 Conv/MatMul weights were found. "
            "Check the selected operators, exclusions, and model precision."
        )
    metadata = {item.key: item.value for item in graph.metadata_props}
    if len(metadata) != len(graph.metadata_props):
        raise ValueError("ONNX metadata contains duplicate keys.")
    interface = deepcopy([*graph.graph.input, *graph.graph.output])
    integer_ops = {"ConvInteger", "MatMulInteger"}
    original_integer_count = sum(node.op_type in integer_ops for node in graph.graph.node)
    with TemporaryDirectory(prefix="onnx-yolo-quant-", dir=destination.parent) as directory:
        staged = Path(directory) / "quantized.onnx"
        ort_quantize_dynamic(
            graph,
            str(staged),
            op_types_to_quantize=operators,
            weight_type=QuantType.QUInt8 if weight_type == "uint8" else QuantType.QInt8,
            per_channel=per_channel,
            reduce_range=reduce_range,
            nodes_to_exclude=excluded,
            use_external_data_format=False,
            extra_options={"MatMulConstBOnly": True},
        )
        converted = onnx.load_model(str(staged))
        quantized_weights = {
            item.name for item in converted.graph.initializer
            if item.data_type in (onnx.TensorProto.INT8, onnx.TensorProto.UINT8)
        }
        converted_count = sum(
            node.op_type in integer_ops
            and len(node.input) > 1 and node.input[1] in quantized_weights
            for node in converted.graph.node
        )
        if converted_count <= original_integer_count:
            raise ValueError("Quantization produced no integer Conv/MatMul weights.")
        converted_interface = [*converted.graph.input, *converted.graph.output]
        if len(interface) != len(converted_interface):
            raise ValueError("Quantization changed the model input/output contract.")
        for original, current in zip(interface, converted_interface, strict=True):
            if original.name != current.name or (
                original.type.tensor_type.elem_type != current.type.tensor_type.elem_type
            ):
                raise ValueError("Quantization changed an input/output name or dtype.")
            before = original.type.tensor_type
            after = current.type.tensor_type
            if before.HasField("shape") and after.HasField("shape"):
                if len(before.shape.dim) != len(after.shape.dim):
                    raise ValueError("Quantization changed an input/output rank.")
                for expected, actual in zip(before.shape.dim, after.shape.dim, strict=True):
                    if expected.HasField("dim_value") and actual.HasField("dim_value"):
                        if expected.dim_value != actual.dim_value:
                            raise ValueError("Quantization changed a fixed input/output dimension.")
            # Shape inference may refine annotations; retain the exported dynamic contract.
            current.CopyFrom(original)
        properties = {item.key: item.value for item in converted.metadata_props}
        properties.update(metadata)
        onnx.helper.set_model_props(converted, properties)
        onnx.checker.check_model(converted)
        onnx.save_model(converted, str(staged))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        try:
            session = ort.InferenceSession(
                str(staged), sess_options=options, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise RuntimeError(
                "The quantized model cannot initialize on CPUExecutionProvider. "
                "Try op_types=('MatMul',) or use a compatible export/runtime."
            ) from exc
        del session
        with staged.open("rb") as reader, destination.open("xb") as writer:
            try:
                copyfileobj(reader, writer)
                writer.flush()
            except BaseException:
                writer.close()
                destination.unlink(missing_ok=True)
                raise
    return destination
