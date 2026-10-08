# onnx-predictor-yolo

为 YOLO、YOLO26 以及兼容的 YOLOE 导出模型提供带类型标注的 ONNX 推理接口。
`YoloPredictor` 支持目标检测、实例分割、旋转框检测、单张图片和批量图片。
`TextPromptEncoder` 和 `VisualPromptEncoder` 为具有提示输入的 YOLOE 模型
生成可复用的提示嵌入。推理结果使用 NumPy 数组表示，坐标对应原始图片。
异步推理支持由调用方管理的线程池；可选的离线动态量化工具可生成
保留类别 metadata 的 8 位 CPU 模型。

英文文档见源码发行包中的 `README.md`；PyPI 项目页面使用英文版本。

## 安装

需要 Python 3.10 或更高版本。

使用 CPU 推理：

```bash
pip install "onnx-predictor-yolo[cpu]"
```

使用 GPU 推理：

```bash
pip install "onnx-predictor-yolo[gpu]"
```

使用 uv 添加项目依赖：

```bash
uv add "onnx-predictor-yolo[cpu]"
```

`cpu` 安装 `onnxruntime`，`gpu` 安装 `onnxruntime-gpu`。同一个环境中只应
安装其中一种运行时。如果应用已经提供兼容的 ONNX Runtime，也可以不选择
这两个可选依赖。使用 CUDA 还需要安装与 GPU 运行时版本匹配的相关库。

核心依赖为 NumPy、无图形界面的 OpenCV 和 PyYAML。本库不会下载模型权重或分词器。
文本分词器由应用提供，可以按需安装其依赖。

## 单张图片推理

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

建议创建一次预测器后重复使用。`model(image)` 与 `model.predict(image)`
等价。每张图片必须为非空的 BGR `uint8` 数组，形状为 `(height, width, 3)`。
RGB、灰度、RGBA 或浮点图片需要先转换。图片读取、可视化和保存由应用处理。

对于输出已经过处理的 YOLO26 模型：

```python
model = YoloPredictor("yolo26n.onnx", output_format="end2end")
result = model(image)
```

请根据实际导出张量选择输出格式。导出原始类别分数的 YOLO26 模型仍使用
`output_format="raw"`。对于已经处理的输出，包括格式兼容的内置 NMS 导出，
本库不会再次执行 NMS。

## 类别 ID 与名称

预测器首先读取 ONNX 的 `names` metadata。类别 ID 在筛选和 NMS 后保留
原始的零起始编号；不会按名称排序，也不会将缺失名称替换为数字字符串。
如果 metadata 缺失、为空或无效，且未显式提供类别名称，构造预测器时就会
抛出 `ValueError`，即使指定了 `num_classes` 也不例外。

对于缺少有效 metadata 的模型，可以按训练时的类别 ID 顺序传入名称列表，
或者提供训练数据集的 YAML 文件：

```python
model = YoloPredictor("custom.onnx", names=["zebra", "ant"], output_format="raw")

# Alternatively, read the names field from the training dataset YAML.
model = YoloPredictor("custom.onnx", names="data.yaml", output_format="raw")
```

YAML 必须包含 `names` 字段，其值可以是列表或 ID 到名称的映射。例如：

```yaml
nc: 2
names:
  0: zebra
  1: ant
```

文件路径支持字符串和 `pathlib.Path`。映射的 ID 必须唯一且从 0 连续编号，
每个名称都必须是非空字符串。若 YAML 包含 `nc`，其值必须等于名称数量。
文件不存在、YAML 无效、键重复或显式名称配置不合法时，都会报错，不会回退。

对于固定类别模型，有效 metadata 与显式配置必须完全一致；冲突时会报错，
不会给模型输出重新贴标签。metadata 无法使用时，可以通过显式列表或 YAML
补充名称。原始输出的类别通道数必须与声明数量一致；已处理输出中，正分数
检测的整数类别 ID 若超出声明范围，也会报错。

`model.names` 和 `result.names` 为只读映射。`result.class_names` 返回
与 `result.class_ids` 一一对应的元组，经过 batch 拆分、筛选和 NMS 后仍然
保持对应关系。请使用实际训练时的 metadata 或数据集配置：库可以校验 ID
及数量，但无法从权重恢复名称含义，也无法识别数量相同却语义错误的类别表。
运行时 YOLOE 提示的类别顺序遵循后文说明。

## 批量推理

```python
images = [cv2.imread(path) for path in ("first.jpg", "second.jpg", "third.jpg")]
if any(item is None for item in images):
    raise ValueError("An input image could not be loaded.")

results = model(images, batch_size=8, conf=0.3)
for source, result in zip(images, results):
    assert result.image_shape == source.shape[:2]
```

- 传入 HWC 数组时，返回一个 `Results` 对象。
- 传入 HWC 图片序列或 BHWC NumPy 数组时，返回 `list[Results]`。
- 图片序列中的图片可以具有不同分辨率。每张图片分别记录坐标变换，
  结果保持输入顺序，并还原到各自的原始尺寸。
- 空序列返回 `[]`，不会执行推理会话。
- `batch_size` 限制每次会话调用处理的真实图片数量。动态 batch 模型使用
  指定的大小；未指定时，一次处理所有输入图片。
- 固定 batch 模型会将图片分组处理。最后一组不足固定大小时，重复最后
  一张图片及其提示数据进行补齐，并丢弃补齐样本的结果。
- batch 固定为 1 的模型会逐张调用。补齐要求模型的不同样本之间相互独立，
  常规 YOLO 模型符合这一条件；包含跨样本运算的模型需要专门的批处理策略。

## 异步推理与线程

`await model.predict_async(...)` 接受 `predict` 的全部参数，包括 batch、
embedding 和每次调用的提示名称，返回类型也保持一致。预处理、会话推理
和后处理均在线程中完成，不会阻塞事件循环。

通过 `executor` 传入 `ThreadPoolExecutor` 可以限制请求并发数；未提供时
使用事件循环的默认线程池。传入的线程池由调用方管理，所有任务结束后再
关闭。以下示例沿用批量推理示例中的 `images`：

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

已有事件循环时，直接 await `infer_many`。单次 batch 请求可以使用
`await model.predict_async(images, batch_size=8, executor=executor)`。
一次调用内部的各个 batch 仍按顺序处理；多个调用按线程池容量并发执行。
`asyncio.gather` 保持请求顺序。处理大型数据流时，应限制待处理请求数量，
不要一次性提交全部图片。

同步应用也可以直接在线程池中复用预测器：

```python
with ThreadPoolExecutor(max_workers=2) as executor:
    results = list(executor.map(model.predict, images))
```

线程池控制请求级并发；ONNX Runtime 的 `SessionOptions.intra_op_num_threads`
控制算子内部并行。应结合设置两者，避免线程争抢 CPU；示例中的线程数并非
经过调优的默认值。使用 `ORT_PARALLEL` 时，`inter_op_num_threads` 控制
计算图中不同算子的并发。传入已有 `InferenceSession` 时，应在创建该会话
时配置这些选项。

异步执行可以改善响应性和并发吞吐，但不意味着单张图片延迟自动降低。
请在目标硬件上评估 batch 和并发设置。若执行提供程序或自定义会话要求
串行调用，包括 CUDA Graph 会话，请使用单线程池。任务进行期间不要修改
输入图片、embedding 或预测器/会话配置。取消异步任务只会停止等待结果，
不会中断已经运行的线程；相关输入必须保持有效，直到线程任务结束。

## 动态量化

安装可选的转换依赖和 CPU 运行时：

```bash
pip install "onnx-predictor-yolo[cpu,quantization]"
```

如果已安装 ONNX Runtime，只需添加 `quantization` extra。该 extra 只添加
ONNX，不会安装第二套运行时。量化是显式执行的离线步骤，与预测器构造分开：

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

请沿用源模型的 `output_format` 和 `task`，例如已处理输出的 YOLO26 导出
使用 `end2end`。量化后仍使用相同的图片预处理、batch 接口、提示输入和
类别名称校验。

`quantize_dynamic` 返回输出文件的 `Path`，支持以下选项：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `op_types` | `("Conv", "MatMul")` | 量化具有常量 FP32 权重的受支持节点 |
| `weight_type` | `"uint8"` | 8 位权重；仅量化 MatMul 时也支持 `int8` |
| `per_channel` | `False` | 仅量化 MatMul 时支持按通道量化权重 |
| `reduce_range` | `False` | 将权重限制为 7 位量化范围 |
| `nodes_to_exclude` | `()` | 保持不量化的计算图节点名称，必须精确匹配 |

为兼容最低支持版本的 ONNX Runtime，Conv 使用无符号、按张量量化的权重；
请求有符号或按通道量化 Conv 时会报错。若仅需量化 MatMul 权重，例如
兼容的编码器模型，可使用：

```python
quantized = quantize_dynamic(
    "encoder.onnx",
    "encoder.quantized.onnx",
    op_types=("MatMul",),
    weight_type="int8",
    per_channel=True,
)
```

只转换符合条件的 Conv/MatMul 权重，其他算子和动态提示 embedding 保持
浮点格式，不需要校准图片。输出必须是已有目录中的新文件，源文件不会
被修改。源模型的外部权重会被读取并合并到单个输出 ONNX 文件，因此结果
需要满足 ONNX 单文件大小限制。

转换器保留原始 metadata，包括 `names`、`task` 和 `imgsz`，并恢复原始
输入输出声明。不会为缺失的类别名称生成默认值；请按类别名称章节的说明
向 `YoloPredictor` 提供训练类别。排除的节点不存在、没有可量化权重或
未产生新的整数算子时都会报错。写出最终文件前，会检查 ONNX 合法性和
CPU 会话能否初始化；这些检查不会运行推理，也不代表精度或速度验证。

8 位权重可以减少存储和内存传输，但不会减少模型原有的运算步骤。运行时
计算激活量化参数也会增加开销，小模型甚至可能因新增图节点而变大。不能
保证延迟更低、整体文件更小或精度不变；部署前应使用代表性数据评估文件
大小、任务精度和延迟。动态量化不等同于 CUDA/TensorRT INT8 导出，本工具
只校验 CPU 加载。CNN 模型可能更适合有校准数据的静态量化，本接口不执行
静态量化。

## 实例分割与旋转框

以下示例沿用前面示例中的 `image` 或 `images`。

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

这两类任务也支持批量输入和格式兼容的原始输出。掩码系数的数量由原型张量
的通道数确定。返回的掩码为原图分辨率的布尔数组。掩码经过缩放、去除填充
并裁剪到还原后的检测框，再进行阈值判断；概率阈值 0.5 对应 logit 阈值 0。

旋转框使用 `[cx, cy, width, height, angle_radians]` 格式。正角度的方向
从 x 轴正方向转向图像 y 轴正方向。旋转角和角点保持原始几何形状，不进行
边界裁剪；`boxes` 则提供裁剪到图像范围内的轴对齐外接框。原始 OBB 输出
的 NMS 使用 OpenCV 的旋转矩形重叠计算。

## YOLOE 推理

### 固定词表模型

类别已经固化到权重中的 YOLOE 导出模型使用常规接口：

```python
model = YoloPredictor("yoloe-fixed-seg.onnx", output_format="raw")
results = model(images, batch_size=4)
```

此类模型没有 embedding 输入，不能在运行时接收新提示。使用后面的动态
提示流程，需要模型实际导出了对应的提示输入。

### 预计算提示嵌入

```python
import numpy as np

model = YoloPredictor(
    "yoloe-prompt-det.onnx", output_format="raw", names=["cat", "dog"]
)
embeddings = np.load("cat-dog-embeddings.npy", allow_pickle=False)
result = model(image, embeddings=embeddings)
results = model(images, embeddings=embeddings, batch_size=4)
```

检测模型必须包含一个 NCHW 图片输入，以及一个形状为 `(B, Q, D)` 的
浮点 embedding 输入，例如 `tpe`、`vpe` 或 `cls_pe`。输入从模型计算图
中识别，不要求 embedding 输入使用固定名称。该接口支持格式兼容的检测
和分割模型，以及原始输出和已经处理的输出。

`embeddings` 支持以下形状：

| 形状 | 含义 |
| --- | --- |
| `(Q, D)` | 所有图片共享同一组提示 |
| `(1, Q, D)` | 所有图片共享同一组提示 |
| `(B, Q, D)` | 按图片输入顺序分别提供提示 |

`Q` 为有效类别数，`D` 为嵌入维度。同一个 embedding 数组中的每张图片
必须具有相同的有效类别数；类别数不同时，应分别调用。类别 ID 对应
embedding 的行顺序，而不是固定的训练类别表。类别名称是必需的，必须与
提示的顺序和数量一致。默认读取 metadata；当前提示与导出时词表不同时，
请显式传入 `names`。只有包含 embedding 输入的模型允许这样更换词表。
同一次调用中的所有图片必须共享相同的类别含义和顺序，即使 embedding 不同。

复用检测器并更换提示时，通过 `prompt_names` 为新 embedding 提供名称。
该参数与 `names` 一样支持列表、映射或 YAML 路径，但只对当前调用生效。
每个结果保存自己的名称映射，`model.names` 仍保留构造时的词表。
预测器在首次调用前，仍必须通过有效 metadata 或构造参数 `names` 获得类别名称。

```python
result = model(
    image,
    embeddings=np.load("bird-embeddings.npy", allow_pickle=False),
    prompt_names=["bird"],
)
print(result.class_names)
```

如果模型的类别容量固定，未使用的位置会自动补零。这些位置不会参与原始
输出的类别选择与 NMS，已经处理的输出也会过滤补零类别。提示数量超过
模型容量时抛出错误，不会静默截断。对于已经处理的输出，补零类别可能在
模型内部占用 top-k 名额；外部过滤无法恢复因此被丢弃的真实检测结果。
动态类别数导出或原始输出导出可以避免这一计算图内部限制。

当模型支持与图片 batch 匹配的提示 batch 时，共享 embedding 会扩展到
相同大小。如果提示输入的 batch 固定为 1，每次调用只传入一份提示，
模型需要支持将其广播到图片 batch。对于逐图提示和 batch 固定为 1 的
提示输入，本库会逐图调用，并按需补齐模型要求的固定图片 batch。

### 文本提示

请使用与 YOLOE 模型配套导出的文本编码器。分词器词表、上下文长度、
预处理、归一化方式和嵌入空间都需要与导出模型匹配；仅嵌入维度相同
并不能保证兼容。

以下示例使用一个可选的分词器依赖：

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

示例中的 `open-clip-torch` 是应用侧的可选依赖，不属于本库核心依赖。
可以替换为任何与模型匹配、返回整数 `(Q, context_length)` NumPy 数组的
可调用对象。也可以不提供分词器，直接将 token 数组传给
`TextPromptEncoder.encode`。

文本编码器应只有一个 int32/int64 二维 token 输入，以及一个二维
`(Q, D)` 输出。固定 token batch 会自动拆分并补齐。返回的 embedding
形状为 `(1, Q, D)`，类型为 float32。可通过 `normalize=True` 显式执行
L2 归一化，默认保持编码器输出。可以编码一次后重复使用或保存结果。

### 视觉提示

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

参考框使用原图中的 `xyxy` 坐标。同一类别的多个区域会取并集，类别 ID
必须从 0 开始连续编号；省略 `class_ids` 时，每个区域视为一个独立类别。
也可以使用 `masks` 代替参考框，其形状为
`(number_of_regions, original_height, original_width)`，数值为 0/1 或
0/255。`boxes` 和 `masks` 必须且只能提供其中一种。

视觉编码器需要一个 RGB NCHW 图片输入、一个浮点
`(B, capacity, mask_height, mask_width)` 提示掩码输入，以及一个
`(B, capacity, D)` embedding 输出。参考区域与图片预处理使用相同的
letterbox 变换。固定掩码尺寸从计算图读取；动态掩码尺寸按
`imgsz / mask_stride` 计算，`mask_stride` 默认为 8。如果某个类别的区域
在掩码分辨率下变为空，会抛出错误。

返回的 `(1, active_classes, D)` 数组不包含未使用的容量位置，可复用于
参考图、其他目标图或目标图片 batch。若每张目标图使用不同参考图，
可以分别编码，并在有效类别数相同的前提下沿第 0 维拼接 embedding。
固定 batch 的视觉编码器会在内部重复参考图，仅返回第一份结果。

## 配置

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

| 构造参数 | 默认值 | 说明 |
| --- | --- | --- |
| `task` | `None` | 优先读取任务元数据，再根据原型输出识别分割，否则按检测处理 |
| `output_format` | `"auto"` | `raw`、`end2end`，或根据形状进行无歧义识别 |
| `layout` | `"channels_first"` | 原始输出的特征轴布局，也支持 `channels_last` |
| `imgsz` | `None` | 正方形边长，或 `(height, width)` |
| `resize` | `"letterbox"` | 居中 letterbox 或 `stretch`；OBB 必须使用 letterbox |
| `pad_value` | `114` | 灰度值或 BGR 元组，各通道为 `[0, 255]` 范围内的整数 |
| `names` | `None` | 训练顺序的名称列表、ID 映射或数据集 YAML 路径；省略时必须有有效 metadata |
| `num_classes` | `None` | 导出类别数；带提示输入时为完整提示容量 |
| `providers` | `None` | 默认使用 CPU，GPU 执行需要显式指定 |
| `session_options` | `None` | 创建新会话时使用的 ONNX Runtime 配置 |
| `image_input` | `None` | 输入识别存在歧义时，显式指定图片输入名称 |

静态高宽从模型读取，显式传入的 `imgsz` 必须与其一致。动态尺寸使用
`imgsz` 元数据，或由调用方显式指定。填充颜色在 BGR 转 RGB 之前应用，
默认是灰色 114；使用 `stretch` 时，`pad_value` 不影响结果。预处理的
FP32/FP16 类型由模型声明决定，后处理统一使用 float32。

模型参数也可以传入已有的 `onnxruntime.InferenceSession`。此时应在
会话本身配置执行提供程序和会话选项。两个提示编码器同样支持已有会话。
请求的执行提供程序不可用或初始化失败时会报告错误，不会静默声明已使用 GPU。

| 推理参数 | 默认值 | 说明 |
| --- | --- | --- |
| `batch_size` | `None` | 每次会话调用处理的真实图片数量上限 |
| `embeddings` | `None` | YOLOE embedding 输入所需的提示数据 |
| `prompt_names` | `None` | 仅 YOLOE：按 embedding 顺序提供当前调用的名称；省略时使用 `model.names` |
| `conf` | `0.25` | 仅保留分数严格大于该阈值的检测 |
| `iou` | `0.45` | 原始输出的 NMS 阈值 |
| `max_det` | `300` | 每张图片的最大检测数量 |
| `classes` | `None` | 允许的类别 ID；`[]` 表示不保留任何类别 |
| `agnostic_nms` | `False` | 是否对原始输出执行跨类别 NMS |
| `mask_threshold` | `0.5` | 掩码的概率阈值 |

原始输出为每个候选选择分数最高的有效类别，默认按类别分别执行 NMS。
已经处理的输出不再次执行 NMS，因此 `iou` 和 `agnostic_nms` 对其无效。
模型内部已经施加的筛选和数量限制无法在外部撤销。

## 张量兼容性

`B` 为实际提交的 batch 大小，`C` 为导出类别数，`K` 为掩码原型通道数，
`N` 为候选框或检测结果数量。

| 任务 | 原始输出 | 已处理输出 | 附加输出 |
| --- | --- | --- | --- |
| 目标检测 | `(B, 4+C, N)` | `(B, N, 6)` | 无 |
| 实例分割 | `(B, 4+C+K, N)` | `(B, N, 6+K)` | `(B, K, Hm, Wm)` |
| 旋转框检测 | `(B, 5+C, N)` | `(B, N, 7)` | 无 |

原始预测行的内容为 `[cx, cy, w, h, class_probabilities..., extras...]`，
不应包含 objectness 通道。附加数据为掩码系数或 OBB 角度。检测和分割的
已处理预测行为 `[x1, y1, x2, y2, score, class_id, mask_coefficients...]`；
OBB 的已处理预测行为 `[cx, cy, w, h, score, class_id, angle_radians]`。
坐标必须采用模型输入画布上的像素单位，而不是归一化坐标。
`layout="channels_last"` 支持通道在最后一维的原始输出，原型张量必须为 NCHW。

自动格式识别依据特征维度、任务附加通道和类别数进行。如果多种格式同时
匹配，需要显式指定 `output_format`。例如，两类别的 `(1, 6, 6)` 张量
存在歧义。没有任务元数据的 OBB 模型还应显式指定 `task="obb"`。
类别名称元数据按字面值解析，不会作为代码执行。

不支持分类、姿态、跟踪、多尺度原始特征头、带 objectness 的 YOLOv5
输出、NHWC 或整数类型的图片输入，以及超出上述图片/embedding 和编码器
约定的任意多输入模型。对于直接接收视觉掩码的检测计算图，需要单独导出
VPE 编码器才能使用此接口。提示编码器与检测器必须配套导出。

## 返回结果

每张图片返回一个 `Results` 数据类。检测结果按置信度降序排列，
经过筛选和 NMS 后，各字段仍保持一一对应。

| 字段 | 形状 / 类型 | 说明 |
| --- | --- | --- |
| `boxes` | `(N, 4)`，float32 | 原图坐标下、已裁剪的 `xyxy` 检测框 |
| `scores` | `(N,)`，float32 | 置信度 |
| `class_ids` | `(N,)`，int64 | 从 0 开始的类别或提示 ID |
| `class_names` | `tuple[str, ...]`，长度为 `N` | 与 `class_ids` 一一对应的名称 |
| `names` | 只读 `Mapping[int, str]` | 当前结果保存的类别名称映射 |
| `image_shape` | `(height, width)` | 原始图片尺寸 |
| `masks` | `(N, height, width)`，bool，或 `None` | 实例掩码 |
| `obb` | `(N, 5)`，float32，或 `None` | 中心坐标、宽高和弧度角 |
| `polygons` | `(N, 4, 2)`，float32，或 `None` | 计算 OBB 四个角点的属性 |

`len(result)` 返回检测数量。空结果保留适用字段的数组维度，其中 `N=0`；
不适用的字段为 `None`。数据类字段不可重新绑定，但数组内容可以修改。
每张图片的完整布尔掩码需要 `N * height * width` 字节，应结合应用内存
预算设置 batch 大小和检测数量上限。
