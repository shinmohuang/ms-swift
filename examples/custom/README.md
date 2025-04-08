# 数据集转换工具

这个工具用于将原始数据集转换为Swift支持的多模态序列分类格式，同时将文本和图像的安全评级和类别转换为数字编码。

## 数据格式说明

### 源数据格式
原始数据集中包含以下字段：
- `id`: 数据项的唯一标识符
- `images`: 图像文件的路径
- `text`: 文本内容
- `text_safety_rating`: 文本安全评级 ("Safe" 或 "Unsafe")
- `text_category`: 文本类别
- `img_safety_rating`: 图像安全评级 ("Safe" 或 "Unsafe")
- `img_category`: 图像类别
- `text_category_scores`: 各文本类别的分数
- `img_category_scores`: 各图像类别的分数

### Swift格式
转换后的Swift多模态序列分类格式包含：
- `messages`: 包含用户角色和内容的消息数组，内容以`<image>`标签开头表示包含图像
- `images`: 图像文件路径的数组
- `id`: 保留原始ID
- `label`: 包含四个数字编码的数组，按顺序为：
  1. `text_safety`: 0 = Safe, 1 = Unsafe
  2. `text_category`: 类别编码，见下文
  3. `img_safety`: 0 = Safe, 1 = Unsafe
  4. `img_category`: 类别编码，见下文

## 类别映射

### 安全评级映射
- Safe: 0
- Unsafe: 1

### 类别映射
- NA: None to Apply: 0
- harassment: 1
- harassment_threatening: 2
- hate: 3
- hate_threatening: 4
- illicit: 5
- illicit_violent: 6
- self_harm: 7
- self_harm_instructions: 8
- self_harm_intent: 9
- sexual: 10
- sexual_minors: 11
- violence: 12
- violence_graphic: 13

对于同时具有多个类别的情况，会从`text_category_scores`或`img_category_scores`中选择分数最高的类别。

转换过程中会在输出目录生成`category_mapping.json`文件，包含完整的类别映射关系，便于后续使用。

## 使用方法

### 数据集转换

运行以下命令转换数据集：

```bash
python convert_dataset.py --input 输入文件路径.json --output 输出文件路径.jsonl
```

例如：

```bash
python convert_dataset.py --input combined_similarity_results_with_moderation.json --output converted_dataset.jsonl
```

### 数据集验证

运行以下命令验证转换后的数据集：

```bash
python test_dataset.py --input 输出文件路径.jsonl --mapping 类别映射文件路径.json
```

例如：

```bash
python test_dataset.py --input output_fixed/converted_dataset.jsonl --mapping output_fixed/category_mapping.json
```

## 安全分类器训练

本项目提供了一个基于Qwen2.5-VL的多任务安全分类器，可同时进行四个分类任务：

1. 文本安全评级：对文本进行安全/不安全的二分类
2. 文本类别识别：对文本进行14个安全类别的多分类
3. 图像安全评级：对图像进行安全/不安全的二分类
4. 图像类别识别：对图像进行14个安全类别的多分类

### 自定义模型结构

模型结构定义在`qwen2_5_vl_safety.py`文件中，主要包括：

- `Qwen2_5_VLWithSafetyClassifiers`类：扩展了Qwen2.5-VL模型，添加了四个分类头
- `get_model_tokenizer_qwen2_5_vl_safety`函数：用于在Swift中加载自定义模型

### 训练命令

使用以下脚本训练安全分类器：

```bash
bash train_safety_classifier.sh
```

该脚本使用Swift的序列分类功能，并通过LoRA微调底层模型，同时训练四个分类头。

### 训练参数调整

可以根据实际需求调整`train_safety_classifier.sh`中的训练参数，主要包括：

- 学习率(`--learning_rate`)
- 批量大小(`--batch_size`, `--eval_batch_size`)
- 训练轮数(`--num_train_epochs`)
- LoRA参数(`--lora_rank`, `--lora_alpha`, `--lora_dropout`)
- 量化位数(`--quantization_bit`)
- 数据集分割比例(`--split_dataset_ratio`)

## Swift训练命令示例

转换后的数据集可以用于Swift的多模态序列分类训练：

```bash
CUDA_VISIBLE_DEVICES=0 swift seq_cls \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset /path/to/converted_dataset.jsonl \
    --train_type lora \
    --output_dir output_seq_cls \
    --num_labels 4 \
    --label_names "text_safety,text_category,img_safety,img_category"
```

注意：需要根据实际需求调整模型、标签数量和其他训练参数。 