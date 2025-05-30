import sys
import os
from swift.llm import (TemplateType, Model, ModelGroup, ModelMeta, register_model, register_dataset, DatasetMeta)
from swift.llm.model.model_arch import MLLMModelArch

# 导入自定义模型和加载函数
from qwen_boxfold_model import (
    QwenBoxModel,
    get_model_tokenizer_qwen_boxfold_model
)


register_dataset(
    DatasetMeta(
        dataset_path="/home/hxm826/ms-swift/examples/custom/qwen_boxfold_ds/qwen_boxfold_dataset.jsonl"
    )
)

# 注册自定义模型
register_model(
    ModelMeta(
        model_type='qwen_boxfold_model',
        model_groups=[
            ModelGroup([
                Model('Qwen/Qwen2.5-VL-3B-Instruct', 'Qwen/Qwen2.5-VL-3B-Instruct'),
                Model('Qwen/Qwen2.5-VL-7B-Instruct', 'Qwen/Qwen2.5-VL-7B-Instruct'),
                Model('Qwen/Qwen2.5-VL-72B-Instruct', 'Qwen/Qwen2.5-VL-72B-Instruct'),
            ]),
        ],
        template=TemplateType.qwen2_5_vl,
        model_arch=MLLMModelArch.qwen2_vl,
        architectures=['QwenBoxModel'],
        requires=['transformers>=4.49', 'qwen_vl_utils>=0.0.6', 'decord'],
        tags=['vision', 'video', 'boxfold'],
        is_multimodal=True,
    ))
