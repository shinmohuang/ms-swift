#!/bin/bash


# 打开调试模式
export SWIFT_DEBUG=1
export TRANSFORMERS_VERBOSITY=info
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 模型和数据集路径
MODEL_PATH="/LOCAL2/hxm826/qwen2.5-vl-safety-model-update"
TRAIN_FILE="/home/hxm826/TensorGuard/dataset/mm-safety-ds/mm_safety_dataset_with_similarity.jsonl"
PROCESSED_DIR="/LOCAL2/hxm826/processed_dataset"
OUTPUT_DIR="/LOCAL2/hxm826/qwen2.5-vl-safety-model-sft-update"

# 注册文件路径 - 修改为包含绝对路径
REGISTER_PATH="/home/hxm826/TensorGuard/model/qwen2_5_vl_safety_register.py"

# 训练参数
NUM_TRAIN_EPOCHS=1
BATCH_SIZE=1
LEARNING_RATE=2e-5
LORA_RANK=8
LORA_ALPHA=16
MAX_LENGTH=1024
GRADIENT_ACCUMULATION_STEPS=1

# 图像预处理（如果需要）
if [ ! -f "${PROCESSED_DIR}/processed_dataset.jsonl" ]; then
    echo "预处理数据集中的图片..."
    mkdir -p $PROCESSED_DIR
    python /home/hxm826/TensorGuard/dataset/mm-safety-ds/preprocess_images.py \
        --dataset $TRAIN_FILE \
        --output_dir $PROCESSED_DIR \
        --max_size 224
    echo "预处理完成!"
fi

# 使用处理后的数据集
PROCESSED_TRAIN_FILE="${PROCESSED_DIR}/processed_dataset.jsonl"

echo "开始使用Swift训练安全分类模型..."
# 设置CUDA内存分配策略，使用正确的键值对格式
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 确保只使用GPU 2，完全避开GPU 0
CUDA_VISIBLE_DEVICES=0,1,2 MAX_PIXELS=32768 swift sft \
    --model $MODEL_PATH \
    --dataset $PROCESSED_TRAIN_FILE \
    --output_dir $OUTPUT_DIR \
    --custom_register_path $REGISTER_PATH \
    --freeze_vit true \
    --torch_dtype bfloat16 \
    --train_type lora \
    --target_modules all-linear text_safety_classifier text_category_classifier image_safety_classifier image_category_classifier\
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --max_length $MAX_LENGTH \
    --logging_steps 10 \
    --save_steps 100 \
    --save_total_limit 3 \
    --seed 42 \
    --freeze_vit true \
    --warmup_ratio 0.05 \
    --report_to wandb \
    --dataloader_num_workers 4 \
    --task_type seq_cls \
    --num_labels 14 \
    --attn_impl flash_attn \
    --use_max_label_value \
    --gradient_checkpointing true

echo "训练完成! 模型保存在: $OUTPUT_DIR"