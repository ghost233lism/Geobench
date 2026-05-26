# geo_data_selector.py 使用文档

`geo_data_selector.py` 用于对图片地理定位数据做自动筛选，流程如下：

1. 递归扫描数据目录中的图片
2. 用一个或多个 VLM 进行采样推理
3. 计算每个采样结果的奖励
4. 按 [caculate_p.md](/nfs/sunboyuan/Geobench/ms-swift/data_clean/caculate_p.md) 计算每张图的保留概率
5. 按指定数量下采样
6. 保留原始目录结构导出筛选后的图片，并保存完整元数据

脚本文件：

- [geo_data_selector.py](/nfs/sunboyuan/Geobench/ms-swift/data_clean/geo_data_selector.py)

## 1. 环境

建议先进入你当前使用的环境：

```bash
source /root/sunboyuan/miniconda3/etc/profile.d/conda.sh
conda activate qwen3.5
```

查看参数：

```bash
python /nfs/sunboyuan/Geobench/ms-swift/data_clean/geo_data_selector.py --help
```

## 2. 支持的输入数据

脚本默认会递归扫描 `--data-dir` 及其子目录下的图片，支持扩展名：

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`
- `.webp`
- `.gif`
- `.tif`
- `.tiff`

GT 支持两种来源。

### 2.1 标注清单文件

通过 `--annotation-file` 传入 `json`、`jsonl` 或 `csv`。

图片路径字段支持自动识别这些常见名字：

- `image`
- `images`
- `image_path`
- `path`
- `file_name`
- `filename`
- `relative_path`

GT 文本字段支持自动识别：

- `gt_text`
- `ground_truth`
- `ground_truth_text`
- `label`
- `answer`
- `solution`
- `location`
- `gt`

GT 经纬度字段支持自动识别：

- 纬度：`gt_latitude` `latitude` `lat` `gt_lat`
- 经度：`gt_longitude` `longitude` `lon` `lng` `gt_lon` `gt_lng`

如果你的字段名不在这些列表里，可以手动指定：

- `--image-field`
- `--gt-text-field`
- `--gt-lat-field`
- `--gt-lon-field`

### 2.2 同名 sidecar json

如果不传 `--annotation-file`，脚本会尝试读取与图片同名的 `.json` 文件。

例如：

```text
data/
  a/b/c.jpg
  a/b/c.json
```

`c.json` 中只要包含 GT 文本，或者 GT 经纬度即可。

## 3. 三种奖励模式

通过 `--reward-mode` 指定。

### 3.1 `text_geocode`

流程：

1. 模型输出自然语言地点描述
2. 用 OpenCage 做地理编码
3. 与 GT 经纬度或 GT 文本地理编码结果比较
4. 用 geoscore 计算奖励

适用场景：

- 你的模型输出的是 `国家; 行政区; 更细位置`
- GT 是地点文本，或者 GT 是经纬度

要求：

- 必须提供 `--opencage-api-keys`

相关参数：

- `--max-distance`
- `--confidence-threshold`

### 3.2 `latlon_geoscore`

流程：

1. 模型直接输出经纬度
2. 若 GT 也是经纬度，则直接比距离
3. 若 GT 只有文本，则先将 GT 文本 geocode 成经纬度
4. 用 geoscore 计算奖励

适用场景：

- 你希望模型直接输出坐标

要求：

- 如果 GT 没有经纬度，只有文本，需要提供 `--opencage-api-keys`

说明：

这个模式现在直接使用独立的经纬度生成提示词文件：
[prompt_latlon_geoscore.txt](/nfs/sunboyuan/Geobench/ms-swift/data_clean/prompt_latlon_geoscore.txt)。

### 3.3 `llm_judge`

流程：

1. 模型输出自然语言地点描述
2. 采样阶段结束后，再统一加载一个评分模型
3. 评分模型比较预测和 GT 文本
4. 输出 `0-1` 的 JSON 分数

适用场景：

- 你不想依赖地理编码 API
- 你希望奖励更偏语义一致性

要求：

- 必须提供 `--judge-model`
- GT 需要有文本字段

对应的 judge 提示词现在拆成两种，可通过参数切换：

- [prompt_judge_banded.txt](/nfs/sunboyuan/Geobench/ms-swift/data_clean/prompt_judge_banded.txt)
- [prompt_judge_exponential.txt](/nfs/sunboyuan/Geobench/ms-swift/data_clean/prompt_judge_exponential.txt)

## 4. 多模型采样策略

你要求的“多个模型不要反复加载”已经按下面方式实现：

1. 加载第一个模型
2. 对所有图片全部采样
3. 立刻把该模型的生成结果写到磁盘
4. 释放模型
5. 加载下一个模型
6. 重复直到所有模型结束
7. 最后再统一进入奖励阶段

因此不会在多个模型之间来回切换加载。

## 5. 双卡和 vLLM

脚本使用 `swift.infer_engine.VllmEngine`。

双卡常用参数：

- `--cuda-visible-devices 0,1`
- `--tensor-parallel-size 2`

常用推理参数：

- `--gpu-memory-utilization`
- `--max-model-len`
- `--max-num-seqs`
- `--infer-batch-size`
- `--temperature`
- `--top-p`
- `--top-k`
- `--max-tokens`

### 5.1 模型路径与自动适配

`--models` 和 `--judge-model` 现在支持两种写法：

- 直接写模型目录绝对路径
- 只写 `/nfs/sunboyuan/model` 下的目录名

例如：

- `Qwen3.5-9B`
- `Qwen3-VL-8B-Instruct`
- `InternVL3_5-8B`
- `/nfs/sunboyuan/model/gemma-3-12b-it`

脚本会自动：

- 在给定目录下查找真正包含 `config.json` 的模型根目录
- 识别模型家族
- 给 `ms-swift` 传合适的 `model_type/template_type`

当前已适配的本地常见家族包括：

- `Qwen3-VL`
- `Qwen3.5`
- `Qwen3`
- `InternVL`
- `Gemma3`

注意：

- 生成模型必须是支持图片输入的多模态模型
- `llm_judge` 的评分模型可以是纯文本模型，也可以是多模态模型，但当前 judge 实际输入是文本，不会再次读图

## 6. 当前主要参数说明

下面只列最常用、最关键的参数。

### 6.1 数据与输入

- `--data-dir`
  图片根目录，脚本会递归扫描
- `--output-dir`
  输出目录，保存中间结果、统计结果和筛选结果
- `--annotation-file`
  可选，支持 `json/jsonl/csv`
- `--image-field`
- `--gt-text-field`
- `--gt-lat-field`
- `--gt-lon-field`
  以上字段名在自动识别不准时手动指定

### 6.2 生成模型相关

- `--models`
  逗号分隔的模型列表
- `--samples-per-model`
  每个模型各采多少个解，长度必须与 `--models` 一致
- `--total-samples`
  如果不写 `--samples-per-model`，则总采样数按模型数自动均分
- `--temperature`
- `--top-p`
- `--top-k`
- `--max-tokens`
  生成模型输出长度上限，当前默认值是 `1024`

### 6.3 vLLM 与显存相关

- `--infer-batch-size`
  每个 batch 处理多少张图片，当前默认值是 `16`
- `--max-model-len`
  当前默认值是 `8192`
- `--max-num-seqs`
  vLLM 内部最大并发序列数，当前默认值是 `128`
- `--gpu-memory-utilization`
  当前默认值是 `0.85`
- `--cuda-visible-devices`
- `--tensor-parallel-size`
- `--disable-custom-all-reduce`
- `--enforce-eager`

### 6.4 奖励相关

- `--reward-mode`
  三选一：`text_geocode`、`latlon_geoscore`、`llm_judge`
- `--judge-model`
  仅 `llm_judge` 时需要
- `--judge-temperature`
  当前默认值是 `0.0`
- `--judge-max-tokens`
  judge 输出长度上限，当前默认值是 `1024`
- `--judge-prompt-mode`
  `banded` 或 `exponential`
- `--disable-model-thinking`
  可选。关闭模型原生 thinking 模式，适用于 InternVL3.5、Qwen3、Qwen3.5 这类带显示思考能力的模板
- `--opencage-api-keys`
  `text_geocode` 必填；`latlon_geoscore` 在 GT 只有文本时需要
- `--max-distance`
- `--confidence-threshold`
- `--prompt-file`
  可选。若不传，脚本会按 reward mode 自动选择独立 prompt 文件：
  [prompt_text_geocode.txt](/nfs/sunboyuan/Geobench/ms-swift/data_clean/prompt_text_geocode.txt)、
  [prompt_latlon_geoscore.txt](/nfs/sunboyuan/Geobench/ms-swift/data_clean/prompt_latlon_geoscore.txt)。
  `llm_judge` 的生成阶段与 `text_geocode` 共用同一个生成 prompt。

### 6.5 概率与下采样相关

- `--max-images`
  最多处理多少张图
- `--keep-count`
  最终保留多少张，不写则默认全保留
- `--mid-target`
- `--tau-mu`
- `--tau-sigma`
- `--epsilon`
- `--variance-mode`
  `exp_std` 或 `percentile`
- `--random-seed`

### 6.6 导出相关

- `--export-mode`
  `symlink`、`copy`、`manifest_only`
- `--selected-root-name`
  导出图片子目录名，默认 `selected_images`
- `--geocode-cache-file`
  可选，自定义 geocode 缓存文件位置

## 7. 保留概率计算

脚本按 [caculate_p.md](/nfs/sunboyuan/Geobench/ms-swift/data_clean/caculate_p.md) 实现：

对每张图，先统计所有采样奖励：

- 均值 `mu_x`
- 标准差 `sigma_x`

然后计算：

- 中等难度分数 `S_mid`
- 区分度分数 `S_var`
- 合成价值 `V(x) = S_mid * S_var`
- 归一化采样概率 `p(x)`

### 7.1 区分度分数两种实现

通过 `--variance-mode` 指定。

`exp_std`：

```text
S_var = 1 - exp(-sigma / (tau_sigma * s_sigma + epsilon))
```

`percentile`：

```text
S_var = sigma 在全体图片标准差中的分位数
```

### 7.2 相关参数

- `--mid-target`
- `--tau-mu`
- `--tau-sigma`
- `--epsilon`

其中：

- 不指定 `--mid-target` 时，默认使用全体图片奖励均值的均值

## 8. 下采样与导出

### 8.1 控制处理数量

- `--max-images`：最多处理多少张图

### 8.2 控制最终保留数量

- `--keep-count`：最终下采样保留多少张
- 如果不指定，则默认全部保留，只计算概率不裁剪

### 8.3 导出方式

通过 `--export-mode` 指定：

- `symlink`：创建软链接，最快，最省空间
- `copy`：直接复制图片
- `manifest_only`：只写元数据，不导出图片

导出图片会保留原有相对目录结构。

## 9. 输出文件说明

所有结果写入 `--output-dir`。

主要文件：

- `run_config.json`
  当前运行配置，也会记录生成模型和 judge 模型的自动识别结果
- `discovered_items.jsonl`
  扫描到并成功匹配 GT 的图片列表
- `generations/<model>.jsonl`
  每个模型自己的采样结果
- `samples.jsonl`
  所有模型汇总后的采样结果与奖励
- `image_stats.jsonl`
  每张图片的奖励均值、标准差、`S_mid`、`S_var`、`value`、`probability`
- `selection.jsonl`
  最终选择结果，每张图片是否被选中
- `summary.json`
  运行摘要
- `geocode_cache.json`
  地理编码缓存

如果启用了图片导出，还会生成：

- `<output-dir>/<selected_root_name>/...`

默认是：

- `<output-dir>/selected_images/...`

## 10. 常用命令示例

### 9.1 自然语言地点输出 + OpenCage geoscore

```bash
source /root/sunboyuan/miniconda3/etc/profile.d/conda.sh
conda activate qwen3.5

python /nfs/sunboyuan/Geobench/ms-swift/data_clean/geo_data_selector.py \
  --data-dir /path/to/images \
  --annotation-file /path/to/annotations.jsonl \
  --output-dir /path/to/output_run_01 \
  --models Qwen3.5-4B,Qwen3.5-9B \
  --total-samples 8 \
  --temperature 0.9 \
  --cuda-visible-devices 0,1 \
  --tensor-parallel-size 2 \
  --reward-mode text_geocode \
  --opencage-api-keys key1,key2,key3,key4 \
  --max-distance 2000 \
  --variance-mode exp_std \
  --keep-count 5000 \
  --export-mode symlink
```

### 9.2 直接输出经纬度 + geoscore

```bash
source /root/sunboyuan/miniconda3/etc/profile.d/conda.sh
conda activate qwen3.5

python /nfs/sunboyuan/Geobench/ms-swift/data_clean/geo_data_selector.py \
  --data-dir /path/to/images \
  --annotation-file /path/to/annotations.jsonl \
  --output-dir /path/to/output_run_latlon \
  --models Qwen3.5-9B \
  --total-samples 8 \
  --temperature 0.8 \
  --cuda-visible-devices 0,1 \
  --tensor-parallel-size 2 \
  --reward-mode latlon_geoscore \
  --opencage-api-keys key1,key2 \
  --max-distance 2000 \
  --keep-count 3000
```

### 9.3 文本输出 + LLM 判分

```bash
source /root/sunboyuan/miniconda3/etc/profile.d/conda.sh
conda activate qwen3.5

python /nfs/sunboyuan/Geobench/ms-swift/data_clean/geo_data_selector.py \
  --data-dir /path/to/images \
  --annotation-file /path/to/annotations.jsonl \
  --output-dir /path/to/output_run_judge \
  --models Qwen3.5-4B,Qwen3.5-9B \
  --samples-per-model 4,4 \
  --temperature 0.9 \
  --cuda-visible-devices 0,1 \
  --tensor-parallel-size 2 \
  --reward-mode llm_judge \
  --judge-model Qwen3.5-9B \
  --judge-temperature 0 \
  --max-tokens 1024 \
  --judge-max-tokens 1024 \
  --variance-mode percentile \
  --keep-count 4000
```

## 11. 标注文件示例

### 10.1 jsonl 示例

```json
{"image_path": "a/b/001.jpg", "gt_text": "United States; California; San Francisco", "gt_latitude": 37.7749, "gt_longitude": -122.4194}
{"image_path": "a/b/002.jpg", "gt_text": "France; Ile-de-France; Paris"}
```

### 10.2 csv 示例

```csv
image_path,gt_text,gt_latitude,gt_longitude
a/b/001.jpg,"United States; California; San Francisco",37.7749,-122.4194
a/b/002.jpg,"France; Ile-de-France; Paris",48.8566,2.3522
```

### 10.3 sidecar json 示例

`001.jpg` 对应 `001.json`：

```json
{
  "gt_text": "Japan; Tokyo; Shibuya",
  "gt_latitude": 35.6595,
  "gt_longitude": 139.7005
}
```

## 12. 参数建议

如果你的目标是做 GRPO 前的数据筛选，比较实用的一组建议是：

- 采样数至少 6 到 8
- 至少使用 1 个较强模型，最好 2 个模型混合采样
- `temperature` 用 `0.7 ~ 1.0`
- `think` 模型建议显式控制 `--max-tokens` 和 `--judge-max-tokens`
- `text_geocode` 模式下优先保证 GT 文本规范
- `llm_judge` 模式下尽量使用比采样模型更强或至少不弱的 judge 模型
- 先用 `manifest_only` 跑一遍分析概率分布，再决定最终 `keep-count`
- 如果你混用不同家族模型，优先直接传模型目录名或绝对路径，让脚本自动适配

## 13. 当前脚本的边界

当前版本已经支持完整主流程，但有几点需要注意：

- `text_geocode` 和 `latlon_geoscore` 使用的是 OpenCage HTTP 接口与本地缓存
- `latlon_geoscore` 模式假设模型能按约束输出可解析的经纬度
- `llm_judge` 模式依赖 judge 模型稳定输出 JSON
- 如果数据目录里有图片但没有可识别 GT，脚本会跳过这些图片
- 生成阶段会检查模型是否支持图片输入，不支持的模型不会被当成 VLM 使用

## 14. 建议运行顺序

推荐按下面顺序试跑：

1. 先用 `--max-images 50`
2. 再用 `--export-mode manifest_only`
3. 检查 `samples.jsonl` 和 `image_stats.jsonl`
4. 确认奖励分布合理
5. 最后扩大到全量数据并设置 `--keep-count`

如果你要，我下一步可以继续补两项内容：

1. 给这份文档再加一份“最小可运行示例”
2. 直接按你的真实数据目录写一条可执行命令
