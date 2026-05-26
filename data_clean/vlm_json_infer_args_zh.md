# `vlm_json_infer.py` 参数说明

本文档详细说明运行 `python ms-swift/data_clean/vlm_json_infer.py` 时可用的所有参数、默认值和作用。

## 一、命令格式

```bash
python ms-swift/data_clean/vlm_json_infer.py \
  --model <模型名> \
  --input-json <输入JSON路径> \
  --prompt-file <prompt文本路径> \
  [其它可选参数...]
```

## 二、参数总览

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `--model` | `str` | 是 | 无 | 选择模型名，只能是以下之一：`Qwen3.5-9B`、`Qwen3.5-27B`、`InternVL3.5-8B`、`InternVL3.5-14B`、`Qwen3-VL-8B`、`Penguin-VL-8B`、`gemma3-12b-it`、`MiniCPM-V-4_5`。 |
| `--model-path` | `str` | 否 | 无 | 模型实际路径。提供后会覆盖 `--model` 的默认路径解析逻辑，适合模型目录不在 `/nfs/sunboyuan/model` 时使用。 |
| `--input-json` | `str` | 是 | 无 | 输入 JSON 文件路径。要求顶层是 `list`，每一项是 `dict`。 |
| `--prompt-file` | `str` | 是 | 无 | prompt 文本文件（txt）路径，作为 `system` 提示词。 |
| `--output-json` | `str` | 否 | `<输入文件名>_<模型字段>_out.json` | 输出 JSON 路径。不填时自动生成到输入 JSON 同目录。 |
| `--image-field` | `str` | 否 | `image` | 输入 JSON 每条样本中，图片路径字段名。 |
| `--image-root` | `str` | 否 | 空（即输入 JSON 所在目录） | 当 `image` 是相对路径时，用于拼接成绝对路径的根目录。 |
| `--gpu` | `str` | 否 | `0` | 设置 `CUDA_VISIBLE_DEVICES`，如 `0`、`1`、`0,1`。 |
| `--batch-size` | `int` | 否 | `8` | 每批推理样本数。 |
| `--batch-timeout-sec` | `float` | 否 | `300.0` | 单批推理超时时间（秒）。`<=0` 表示不启用超时。超时后会中断后续推理。 |
| `--max-tokens` | `int` | 否 | `2048` | 单条生成最大 token 数。 |
| `--temperature` | `float` | 否 | `0.0` | 采样温度。越低越稳定，越高越随机。 |
| `--top-p` | `float` | 否 | `1.0` | nucleus sampling 参数。 |
| `--top-k` | `int` | 否 | `-1` | top-k 参数，`-1` 一般表示不限制。 |
| `--max-model-len` | `int` | 否 | `8192` | 模型最大上下文长度（vLLM 相关）。 |
| `--max-num-seqs` | `int` | 否 | `128` | vLLM 并发序列上限。 |
| `--gpu-memory-utilization` | `float` | 否 | `0.85` | vLLM 显存利用率目标。 |
| `--tensor-parallel-size` | `int` | 否 | `1` | 张量并行数（多卡时可调大）。 |
| `--disable-model-thinking` | flag | 否 | `False` | 加上该参数后，尝试禁用支持该能力模型的 thinking 模式。 |
| `--log-file` | `str` | 否 | `<output_json同名>.log` | 日志文件路径。 |
| `-h`, `--help` | flag | 否 | 无 | 打印帮助信息并退出。 |

## 三、输入输出行为说明

### 1) 输入 JSON 格式要求

- 顶层必须是列表：`[{}, {}, ...]`
- 每项必须是字典
- 默认读取图片字段 `image`，可用 `--image-field` 修改

示例：

```json
[
  {
    "id": 1,
    "image": "images/a.jpg"
  },
  {
    "id": 2,
    "image": "/abs/path/to/b.png"
  }
]
```

### 2) 推理输入组织方式

脚本内部按如下形式构建请求（与 `geo_data_selector.py` 一致）：

- `messages`:
  - `system`: prompt 文件内容
  - `user`: `"Locate this image."`
- `images`: `[图片路径]`

### 3) 输出新增字段

每条样本都会新增两个字段（按模型名自动生成前缀）：

- `xxx_answer`: 原始模型输出文本
- `xxx_inference_completed`: 该条是否成功完成推理（`true/false`）

示例（`--model Qwen3.5-9B`）：

- `qwen3.5_9b_answer`
- `qwen3.5_9b_inference_completed`

## 四、常用命令示例

### 示例 1：最小必填

```bash
python ms-swift/data_clean/vlm_json_infer.py \
  --model Qwen3.5-9B \
  --input-json /path/to/input.json \
  --prompt-file /nfs/sunboyuan/Geobench/ms-swift/data_clean/prompt_text_geocode.txt
```

### 示例 2：指定 GPU、批大小、超时和输出路径

```bash
python ms-swift/data_clean/vlm_json_infer.py \
  --model Qwen3-VL-8B \
  --input-json /path/to/input.json \
  --prompt-file /nfs/sunboyuan/Geobench/ms-swift/data_clean/prompt_text_geocode.txt \
  --gpu 1 \
  --batch-size 16 \
  --batch-timeout-sec 180 \
  --output-json /path/to/output.json \
  --log-file /path/to/output.log
```

### 示例 3：模型不在默认目录，手动指定路径

```bash
python ms-swift/data_clean/vlm_json_infer.py \
  --model MiniCPM-V-4_5 \
  --model-path /custom/model/MiniCPM-V-4_5 \
  --input-json /path/to/input.json \
  --prompt-file /path/to/prompt.txt
```

### 示例 4：相对图片路径配合 `--image-root`

```bash
python ms-swift/data_clean/vlm_json_infer.py \
  --model InternVL3.5-8B \
  --input-json /path/to/meta/data.json \
  --image-root /path/to/dataset_root \
  --prompt-file /path/to/prompt.txt
```

## 五、调参建议

- 显存不足时优先减小：`--batch-size`、`--max-num-seqs`、`--gpu-memory-utilization`
- 结果过于随机时：降低 `--temperature`（例如 `0.0` 到 `0.2`）
- 批次容易卡住时：设置 `--batch-timeout-sec`（如 `120` 或 `180`）
- 多卡推理时：设置 `--gpu 0,1,...` 并将 `--tensor-parallel-size` 与卡数匹配

