# onnx-predictor-yolo

Typed ONNX inference for YOLO, YOLO26, and compatible YOLOE exports.
`YoloPredictor` supports detection, instance segmentation, oriented bounding
boxes, single images, and image batches. `TextPromptEncoder` and
`VisualPromptEncoder` produce reusable embeddings for YOLOE graphs that expose
a prompt input. Results are NumPy arrays in original image coordinates.
Async inference supports caller-controlled thread pools. An optional offline
dynamic quantizer creates 8-bit CPU models while preserving class metadata.

Chinese documentation: `README.zh-CN.md` in the source distribution.

## Installation

Python 3.10 or newer is required.

For CPU inference:

```bash
pip install "onnx-predictor-yolo[cpu]"
```

For GPU inference:

```bash
pip install "onnx-predictor-yolo[gpu]"
```

With uv, add the package to your project:

```bash
uv add "onnx-predictor-yolo[cpu]"
```

Choose `cpu` for `onnxruntime` or `gpu` for `onnxruntime-gpu`. Install only one
runtime distribution in an environment. Installing without an extra lets an
application supply its own compatible ONNX Runtime distribution. CUDA also
requires libraries compatible with the installed GPU runtime.

Core dependencies are NumPy, headless OpenCV, and PyYAML. No weights or tokenizers are
downloaded. A text tokenizer is supplied by the application and can have its
own optional dependencies.

## Single-image inference

```python
import cv2

from onnx_predictor_yolo import YoloPredictor

image = cv2.imread("image.jpg")
if image is None:
    raise FileNotFoundError("image.jpg")

model = YoloPredictor("yolov8n.onnx", output_format="raw")
result = model(image, conf=0.25, iou=0.45)

for box, score, class_id in zip(result.boxes, result.scores, result.class_ids):
    label = result.names[int(class_id)]
    print(label, float(score), box.tolist())
```

Create the predictor once and reuse it. `model(image)` and
`model.predict(image)` are equivalent. Each image must be a nonempty BGR
`uint8` array shaped `(height, width, 3)`. Convert RGB, grayscale, RGBA, or
floating-point images before calling the predictor. Image loading, rendering,
and saving belong to the application.

For processed YOLO26 outputs:

```python
model = YoloPredictor("yolo26n.onnx", output_format="end2end")
result = model(image)
```

Select the format from the exported tensors. A YOLO26 raw export still uses
`output_format="raw"`. Processed outputs, including compatible embedded-NMS
exports, are not subjected to another NMS pass.

## Class IDs and names

The predictor first reads the ONNX `names` metadata. Class IDs keep their
original zero-based values through filtering and NMS; labels are never sorted
alphabetically or replaced with numeric strings. If metadata is missing,
empty, or invalid and no explicit names are provided, construction raises
`ValueError`, even if `num_classes` is set.

For an export without usable metadata, supply the original training labels
in class ID order or the training dataset YAML:

```python
model = YoloPredictor("custom.onnx", names=["zebra", "ant"], output_format="raw")

# Alternatively, read the names field from the training dataset YAML.
model = YoloPredictor("custom.onnx", names="data.yaml", output_format="raw")
```

The YAML must contain `names` as a list or an ID-to-label mapping. For example:

```yaml
nc: 2
names:
  0: zebra
  1: ant
```

String paths and `pathlib.Path` are accepted. Mappings must use unique,
contiguous IDs starting at zero; every label must be a nonempty string. An
optional YAML `nc` must equal the number of labels. Missing files, invalid YAML,
duplicate keys, and invalid explicit labels raise errors without fallback.

For fixed-vocabulary models, valid metadata and explicit names must agree
exactly; conflicting names raise an error instead of relabeling detections.
An explicit list or YAML can supply labels when metadata is unusable. Raw class
channels must match the declared count; processed detections with positive
scores and out-of-range integer class IDs raise an error.

`model.names` and `result.names` are read-only mappings.
`result.class_names` returns a tuple aligned with `result.class_ids`, including
after batch splitting, filtering, and NMS. Use the actual training metadata or
dataset configuration: the library can validate IDs and counts, but cannot
recover label meanings from weights or detect an incorrect same-length
vocabulary. Runtime YOLOE prompts use the prompt order described below.

## Batch inference

```python
images = [cv2.imread(path) for path in ("first.jpg", "second.jpg", "third.jpg")]
if any(item is None for item in images):
    raise ValueError("An input image could not be loaded.")

results = model(images, batch_size=8, conf=0.3)
for source, result in zip(images, results):
    assert result.image_shape == source.shape[:2]
```

- An HWC array returns one `Results` object.
- A sequence of HWC arrays or a BHWC NumPy array returns `list[Results]`.
- Images in a sequence may have different resolutions. Each image gets its
  own transform; returned results preserve input order and source dimensions.
- An empty sequence returns `[]` without running the session.
- `batch_size` limits real images per session call. A dynamic-batch graph uses
  the requested size, or all supplied images when the argument is omitted.
- A fixed-batch graph accepts arbitrary list lengths by splitting the input
  into groups. A short group repeats its final image and associated prompt
  data to meet the fixed dimension; padded results are discarded.
- A batch-one graph therefore processes a list through successive calls.
  Padding assumes the exported model processes samples independently, as
  ordinary YOLO graphs do. Models with cross-sample operations require their
  own batching policy.

## Async inference and threads

`await model.predict_async(...)` accepts every `predict` option, including
batching, embeddings, and per-call prompt names, and returns the same result
type. Preprocessing, session execution, and postprocessing run in a worker
thread so they do not block the event loop.

Pass a `ThreadPoolExecutor` through `executor` to bound concurrent requests.
Without it, the event loop's default executor is used. The caller owns a
supplied pool and shuts it down after outstanding work finishes. For example,
using `images` from the batch example:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

import onnxruntime as ort

from onnx_predictor_yolo import YoloPredictor

options = ort.SessionOptions()
options.intra_op_num_threads = 2
model = YoloPredictor(
    "model.onnx", output_format="raw", session_options=options
)


async def infer_many(images, executor):
    return await asyncio.gather(
        *(model.predict_async(image, executor=executor) for image in images)
    )


with ThreadPoolExecutor(max_workers=2) as executor:
    results = asyncio.run(infer_many(images, executor))
```

Inside an existing event loop, await `infer_many` directly. For one batched
request, use `await model.predict_async(images, batch_size=8, executor=executor)`.
One such call processes its batches sequentially; concurrent calls run in
parallel up to the executor limit. `asyncio.gather` preserves request order.
For large streams, apply backpressure instead of scheduling every image at once.

Synchronous applications can use the same predictor with threads directly:

```python
with ThreadPoolExecutor(max_workers=2) as executor:
    results = list(executor.map(model.predict, images))
```

The executor limits request-level concurrency. ONNX Runtime's
`SessionOptions.intra_op_num_threads` controls parallel work inside operators.
Configure both together to avoid CPU oversubscription; the values above are
examples, not tuned defaults. If using `ORT_PARALLEL`,
`inter_op_num_threads` controls concurrency between graph operators. For an
existing `InferenceSession`, configure these options when creating that session.

Async execution can improve responsiveness and concurrent throughput; it does
not inherently reduce single-image latency. Measure batching and concurrency
on the target hardware. Use a single-worker executor when the execution
provider or custom session requires serialized calls, including CUDA Graph
sessions. Do not mutate input arrays, embeddings, or predictor/session settings
while work is in flight. Cancellation stops awaiting a result but does not
terminate an already running worker; its inputs must remain valid until it ends.

## Dynamic quantization

Install the optional conversion dependency together with a CPU runtime:

```bash
pip install "onnx-predictor-yolo[cpu,quantization]"
```

If an ONNX Runtime distribution is already installed, add only the
`quantization` extra. It adds ONNX without installing a second runtime.
Conversion is an explicit offline step, separate from predictor construction:

```python
from pathlib import Path

from onnx_predictor_yolo import YoloPredictor, quantize_dynamic

source = Path("model.onnx")
quantized = quantize_dynamic(source, "model.quantized.onnx")
print("Source bytes:", source.stat().st_size)
print("Quantized bytes:", quantized.stat().st_size)

model = YoloPredictor(quantized, output_format="raw")
result = model(image)
```

Use the source model's `output_format` and `task`, including `end2end` for a
processed YOLO26 export. The quantized model uses the same image preprocessing,
batch interface, prompt inputs, and class-name validation as its source.

`quantize_dynamic` returns the output `Path`. Its options are:

| Option | Default | Meaning |
| --- | --- | --- |
| `op_types` | `("Conv", "MatMul")` | Quantize supported nodes with constant FP32 weights |
| `weight_type` | `"uint8"` | 8-bit weights; `int8` is also accepted for MatMul-only conversion |
| `per_channel` | `False` | Per-channel weights for MatMul-only conversion |
| `reduce_range` | `False` | Restrict weights to a 7-bit quantization range |
| `nodes_to_exclude` | `()` | Exact graph node names to keep unquantized |

Conv uses unsigned, per-tensor weights for compatibility with the minimum
supported ONNX Runtime version. Signed or per-channel Conv requests raise an
error. To quantize only MatMul weights, for example in a compatible encoder:

```python
quantized = quantize_dynamic(
    "encoder.onnx",
    "encoder.quantized.onnx",
    op_types=("MatMul",),
    weight_type="int8",
    per_channel=True,
)
```

Only eligible Conv/MatMul weights are converted; other operators and dynamic
prompt embeddings stay floating point. No calibration images are required.
The output must be a new file in an existing directory. Source files are kept
unchanged. External source weights are loaded and embedded in one output ONNX
file, so the result must fit the single-file ONNX size limit.

The converter preserves original metadata, including `names`, `task`, and
`imgsz`, and restores the original input/output declarations. Missing class
names are not synthesized: provide training labels to `YoloPredictor` as
described in the class-name section. Unknown exclusions, no eligible weights,
or conversion without new integer operators raise errors. Before publishing
the output, the converter checks ONNX validity and CPU session initialization.
These checks do not run inference or establish accuracy or speed.

8-bit weights can reduce storage and memory traffic, but quantization does not
remove the underlying model operations. Runtime activation quantization adds
overhead, and small models may even grow due to extra graph nodes. Lower latency,
smaller total size, and unchanged accuracy are not guaranteed. Validate model
size, task accuracy, and latency on representative inputs before deployment.
Dynamic quantization is not a CUDA/TensorRT INT8 export; the helper validates
CPU loading only. CNN models may benefit more from calibrated static
quantization, which this helper does not perform.

## Segmentation and oriented boxes

The following examples reuse `image` or `images` from the examples above.

```python
segmenter = YoloPredictor(
    "yolo26n-seg.onnx", task="segment", output_format="end2end"
)
result = segmenter(image, mask_threshold=0.5)
print(result.masks.shape)

obb_model = YoloPredictor(
    "yolo26n-obb.onnx", task="obb", output_format="end2end"
)
result = obb_model(image)
print(result.obb)
print(result.polygons)
```

Both tasks also accept batches and compatible raw exports. Mask coefficients
are inferred from the prototype channels. Masks are boolean arrays at the
original resolution. A probability threshold of 0.5 is applied as logit zero
after resizing, removing padding, and cropping to the restored box.

Oriented boxes contain `[cx, cy, width, height, angle_radians]`. Positive
rotation moves from the positive x axis toward the positive image y axis.
Angles and corners are preserved without clipping; `boxes` contains clipped
axis-aligned envelopes. Raw OBB NMS uses OpenCV rotated rectangle overlap.

## YOLOE inference

### Fixed-vocabulary exports

YOLOE exports with classes baked into their weights use the regular interface:

```python
model = YoloPredictor("yoloe-fixed-seg.onnx", output_format="raw")
results = model(images, batch_size=4)
```

Such graphs have no embedding input and cannot accept new prompts at runtime.
The artifact must expose a prompt input to use the following workflows.

### Precomputed embeddings

```python
import numpy as np

model = YoloPredictor(
    "yoloe-prompt-det.onnx", output_format="raw", names=["cat", "dog"]
)
embeddings = np.load("cat-dog-embeddings.npy", allow_pickle=False)
result = model(image, embeddings=embeddings)
results = model(images, embeddings=embeddings, batch_size=4)
```

The detector must expose one NCHW image input and one floating-point embedding
input shaped `(B, Q, D)`, such as `tpe`, `vpe`, or `cls_pe`. The input is selected
from the graph, without requiring a specific embedding input name. The same
interface supports compatible detection and segmentation heads, in raw or
processed form.

`embeddings` accepts:

| Shape | Meaning |
| --- | --- |
| `(Q, D)` | Shared prompts for every image |
| `(1, Q, D)` | Shared prompts for every image |
| `(B, Q, D)` | Separate prompts for each supplied image, in input order |

`Q` is the active class count and `D` the embedding width. Every sample in an
embedding array has the same active class count. For different class counts,
use separate calls. Class IDs refer to embedding row order, rather than a fixed
training vocabulary. Class names are required and must match that order and
count. Metadata is read by default; explicitly pass `names` when the current
prompts differ from the exported vocabulary. Only models with an embedding
input allow this vocabulary change. All images in one call must share the
same label meanings and order, even when their embeddings differ.

When reusing a detector with new prompts, pass `prompt_names` together with
the new embeddings. It accepts the same list, mapping, or YAML path as `names`
and applies only to that call. Each result retains its own label mapping;
`model.names` remains the constructor vocabulary. A predictor still requires
usable metadata or explicit constructor `names` before its first call.

```python
result = model(
    image,
    embeddings=np.load("bird-embeddings.npy", allow_pickle=False),
    prompt_names=["bird"],
)
print(result.class_names)
```

Unused fixed-capacity class slots are zero-padded automatically. Padded slots
are excluded before raw class selection and NMS, or filtered from processed
results. Too many prompts raise an error; prompts are never silently truncated.
For processed graphs, padded classes can already consume internal top-k slots;
filtering cannot recover discarded real detections. A dynamic-class or raw
export avoids that graph-level limitation.

Shared embeddings are expanded to the image batch when the graph accepts a
matching prompt batch. If the prompt input is fixed at batch one, it is fed
once per call; the graph must support broadcasting it to its image batch.
Per-image embeddings with a batch-one prompt input are handled through
individual calls, with image padding if required by a fixed image batch.

### Text prompts

Use a text encoder exported for the same YOLOE model. The tokenizer vocabulary,
context length, preprocessing, normalization, and embedding space must match
that export; equal embedding width alone does not establish compatibility.

The following example uses an optional tokenizer dependency:

```bash
pip install open-clip-torch
```

```python
import open_clip

from onnx_predictor_yolo import TextPromptEncoder, YoloPredictor

# Use the tokenizer associated with the supplied MobileCLIP export.
clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")
encoder = TextPromptEncoder(
    "mobileclip.onnx",
    tokenizer=lambda texts: clip_tokenizer(texts).cpu().numpy(),
)
labels = ["cat", "dog"]
embeddings = encoder(labels)

model = YoloPredictor("yoloe-text-det.onnx", output_format="raw", names=labels)
results = model(images, embeddings=embeddings, batch_size=4)
```

The example tokenizer is an optional application dependency
(`open-clip-torch`), not a core library dependency. Any matching callable that
returns an integer `(Q, context_length)` NumPy array can be used. Alternatively,
pass a token array directly to `TextPromptEncoder.encode` without a tokenizer.

The text encoder graph must have one int32/int64 rank-two token input and one
rank-two `(Q, D)` output. Fixed token batch sizes are handled by splitting and
padding. The returned embeddings have shape `(1, Q, D)` and dtype float32.
`normalize=True` optionally applies L2 normalization; the default preserves the
encoder's output. Encode once and reuse or save the result across image calls.

### Visual prompts

```python
from onnx_predictor_yolo import VisualPromptEncoder, YoloPredictor

reference = cv2.imread("reference.jpg")
if reference is None:
    raise FileNotFoundError("reference.jpg")

encoder = VisualPromptEncoder("yoloe-vpe.onnx", pad_value=114)
embeddings = encoder(
    reference,
    boxes=[[20, 30, 100, 140], [120, 40, 180, 110]],
    class_ids=[0, 0],
)
model = YoloPredictor(
    "yoloe-visual-det.onnx", output_format="raw", names=["part"]
)
results = model(images, embeddings=embeddings, batch_size=4)
```

Boxes use original-image `xyxy` coordinates. Multiple regions with the same
class ID are combined by union. IDs must be contiguous from zero; omitting
`class_ids` makes every region a separate class. Instead of boxes, provide
`masks` shaped `(number_of_regions, original_height, original_width)` with
values 0/1 or 0/255. Supply exactly one of boxes and masks.

The encoder must expose an RGB NCHW image input, a floating-point
`(B, capacity, mask_height, mask_width)` prompt mask input, and a
`(B, capacity, D)` embedding output. Image preprocessing and reference regions
share the same letterbox transform. Fixed mask dimensions come from the graph;
dynamic dimensions use `imgsz / mask_stride`, with `mask_stride=8` by default.
A class whose regions disappear at the mask resolution raises an error.

The returned `(1, active_classes, D)` array excludes unused capacity slots and
can be reused for the reference image, other target images, or target batches.
For different references per target, encode each reference and concatenate
those arrays along axis zero, using the same class count. Fixed-batch visual
encoders repeat the reference internally and return only its first result.

## Configuration

```python
model = YoloPredictor(
    "dynamic.onnx",
    task="detect",
    output_format="raw",
    imgsz=(384, 640),
    pad_value=(32, 64, 128),
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
```

| Constructor option | Default | Meaning |
| --- | --- | --- |
| `task` | `None` | Use task metadata, then prototype presence; otherwise detection |
| `output_format` | `"auto"` | `raw`, `end2end`, or unambiguous shape-based selection |
| `layout` | `"channels_first"` | Raw output feature axis; `channels_last` also accepted |
| `imgsz` | `None` | Integer square size or `(height, width)` |
| `resize` | `"letterbox"` | Centered letterbox or `stretch`; OBB requires letterbox |
| `pad_value` | `114` | Gray level or BGR tuple, integer channels in `[0, 255]` |
| `names` | `None` | Training-order labels, ID mapping, or dataset YAML path; otherwise metadata is required |
| `num_classes` | `None` | Exported class count; for prompt graphs, full prompt capacity |
| `providers` | `None` | Defaults to CPU; GPU execution must be requested explicitly |
| `session_options` | `None` | ONNX Runtime session options for a newly created session |
| `image_input` | `None` | Explicit image input name for ambiguous graphs |

Static spatial dimensions come from the graph. An explicit `imgsz` must match
them. Dynamic dimensions use `imgsz` metadata or require an explicit size.
Padding is applied in BGR before conversion to RGB, with gray 114 by default.
`pad_value` has no effect for `stretch`. FP32/FP16 preprocessing follows the
declared input dtype; postprocessing uses float32.

The model argument also accepts an existing `onnxruntime.InferenceSession`.
In that case configure providers and session options on the session itself.
This is also supported by both prompt encoders. Unavailable or failed requested
providers are reported instead of silently claiming GPU execution.

| Prediction option | Default | Meaning |
| --- | --- | --- |
| `batch_size` | `None` | Maximum real images per session call |
| `embeddings` | `None` | Prompt data for a YOLOE embedding input |
| `prompt_names` | `None` | YOLOE-only labels for this call, in embedding order; otherwise use `model.names` |
| `conf` | `0.25` | Keep scores strictly greater than this threshold |
| `iou` | `0.45` | Raw-output NMS threshold |
| `max_det` | `300` | Maximum detections per image |
| `classes` | `None` | Allowed class IDs; `[]` keeps none |
| `agnostic_nms` | `False` | Suppress across classes for raw outputs |
| `mask_threshold` | `0.5` | Probability threshold for masks |

Raw predictions select the best active class for each candidate and apply
class-aware NMS by default. Processed exports skip NMS, so `iou` and
`agnostic_nms` do not affect them. Internal export limits cannot be undone.

## Tensor compatibility

`B` is the submitted batch, `C` the exported class count, `K` the prototype
channel count, and `N` the candidate or detection count.

| Task | Raw output | Processed output | Additional output |
| --- | --- | --- | --- |
| Detection | `(B, 4+C, N)` | `(B, N, 6)` | None |
| Segmentation | `(B, 4+C+K, N)` | `(B, N, 6+K)` | `(B, K, Hm, Wm)` |
| OBB | `(B, 5+C, N)` | `(B, N, 7)` | None |

Raw rows are `[cx, cy, w, h, class_probabilities..., extras...]`. They must not
contain an objectness channel. Extras are mask coefficients or an OBB angle.
Processed detection/segmentation rows contain
`[x1, y1, x2, y2, score, class_id, mask_coefficients...]`; processed OBB rows
contain `[cx, cy, w, h, score, class_id, angle_radians]`. Coordinates must use
model-input pixels, not normalized values. Raw channels-last exports are
accepted with `layout="channels_last"`. Prototypes must be NCHW.

Auto format selection uses feature dimensions, task extras, and class count.
If multiple protocols match, specify `output_format` explicitly. For example,
a two-class `(1, 6, 6)` tensor is ambiguous. For metadata-free OBB models, also
specify `task="obb"`. Names metadata is parsed as literals without execution.

Not supported: classification, pose, tracking, raw multi-scale feature heads,
objectness-bearing YOLOv5 outputs, NHWC or integer image inputs, and arbitrary
multi-input graphs outside the documented image/embedding and encoder contracts.
Direct visual-mask detector graphs need a separate VPE encoder export for this
API. Prompt encoders and detectors must be exported as a compatible set.

## Results

Each image produces a `Results` dataclass. Detections are ordered by descending
confidence. Arrays stay aligned after filtering and NMS.

| Field | Shape / type | Meaning |
| --- | --- | --- |
| `boxes` | `(N, 4)`, float32 | Clipped original-image `xyxy` boxes |
| `scores` | `(N,)`, float32 | Confidence scores |
| `class_ids` | `(N,)`, int64 | Zero-based class or prompt IDs |
| `class_names` | `tuple[str, ...]`, length `N` | Labels aligned with `class_ids` |
| `names` | Read-only `Mapping[int, str]` | Class vocabulary captured for this result |
| `image_shape` | `(height, width)` | Original image size |
| `masks` | `(N, height, width)`, bool, or `None` | Instance masks |
| `obb` | `(N, 5)`, float32, or `None` | Center, size, angle in radians |
| `polygons` | `(N, 4, 2)`, float32, or `None` | Property computing OBB corners |

`len(result)` returns the detection count. Empty results retain all applicable
array dimensions with `N=0`. Non-applicable fields are `None`. Dataclass fields
cannot be rebound; arrays remain mutable. Full-resolution boolean masks require
`N * height * width` bytes per image, so select batch and detection limits with
the application memory budget in mind.
