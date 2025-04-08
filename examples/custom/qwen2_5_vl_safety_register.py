import sys
import os
from swift.llm import (TemplateType, Model, ModelGroup, ModelMeta, register_model, register_dataset, DatasetMeta)
from swift.llm.model.model_arch import MLLMModelArch

# 导入自定义模型和加载函数
from qwen2_5_vl_safety import (
    Qwen2_5_VLWithSafetyClassifiers,
    get_model_tokenizer_qwen2_5_vl_safety
)


register_dataset(
    DatasetMeta(
        dataset_path="/home/hxm826/ms-swift/examples/custom/mm-safety-ds/mm_safety_dataset.jsonl"
    )
)

# 注册自定义模型
register_model(
    ModelMeta(
        model_type='qwen2_5_vl_safety',
        model_groups=[
            ModelGroup([
                Model('Qwen/Qwen2.5-VL-3B-Instruct', 'Qwen/Qwen2.5-VL-3B-Instruct'),
                Model('Qwen/Qwen2.5-VL-7B-Instruct', 'Qwen/Qwen2.5-VL-7B-Instruct'),
                Model('Qwen/Qwen2.5-VL-72B-Instruct', 'Qwen/Qwen2.5-VL-72B-Instruct'),
            ]),
        ],
        template=TemplateType.qwen2_5_vl,
        model_arch=MLLMModelArch.qwen2_vl,
        architectures=['Qwen2_5_VLWithSafetyClassifiers'],
        requires=['transformers>=4.49', 'qwen_vl_utils>=0.0.6', 'decord'],
        get_function=get_model_tokenizer_qwen2_5_vl_safety,
        tags=['vision', 'video', 'safety'],
        is_multimodal=True,
    ))
