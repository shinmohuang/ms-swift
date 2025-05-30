# 用于评估带安全分类器的Qwen2.5-VL模型
import torch
import json
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, AutoConfig
from qwen2_5_vl_safety import Qwen2_5_VLWithSafetyClassifiers, get_model_tokenizer_qwen2_5_vl_safety, SAFETY_CATEGORIES
from swift.llm import EvalArguments, eval_main
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import traceback  # 添加此导入以获取详细的错误信息
from qwen_vl_utils import process_vision_info
import datetime
import time


# 设置模型路径 - 请替换为您实际的模型路径
model_dir = "/LOCAL2/hxm826/qwen2.5-safety-model-merged"

# 添加CUDA设备检查和选择逻辑
print("CUDA是否可用:", torch.cuda.is_available())
print("可用的CUDA设备数量:", torch.cuda.device_count())
print("当前CUDA设备:", torch.cuda.current_device())
print("CUDA设备名称:", torch.cuda.get_device_name(0))

# 选择设备
if torch.cuda.is_available():
    try:
        # 获取GPU使用情况
        gpu_memory = []
        for i in range(torch.cuda.device_count()):
            try:
                with torch.cuda.device(i):
                    info = torch.cuda.get_device_properties(i)
                    total_memory = info.total_memory / 1024**2  # 转换为MB
                    allocated_memory = torch.cuda.memory_allocated() / 1024**2
                    free_memory = total_memory - allocated_memory
                    gpu_memory.append((i, free_memory))
                    print(f"GPU {i} - 总内存: {total_memory:.2f}MB, 已用: {allocated_memory:.2f}MB, 可用: {free_memory:.2f}MB")
            except Exception as e:
                print(f"无法获取GPU {i}的信息: {e}")
                continue

        if not gpu_memory:
            print("没有可用的GPU，切换到CPU")
            device = 'cpu'
        else:
            # 选择内存最多的GPU
            device_id, free_mem = max(gpu_memory, key=lambda x: x[1])
            if free_mem < 1024:  # 如果最大可用内存小于1GB
                print(f"警告：GPU {device_id}可用内存较低 ({free_mem:.2f}MB)")
            device = f'cuda:{device_id}'
            print(f"自动选择GPU {device_id}，可用内存: {free_mem:.2f}MB")

            # 设置内存分配器
            torch.cuda.set_per_process_memory_fraction(0.95, device_id)  # 限制GPU内存使用为95%
            torch.cuda.empty_cache()  # 清空GPU缓存
    except Exception as e:
        print(f"GPU选择过程出错: {e}")
        print("切换到CPU模式")
        device = 'cpu'
else:
    device = 'cpu'
    print("CUDA不可用，使用CPU")

print(f"最终使用设备: {device}")

# 1. 获取模型和处理器
print("开始加载模型...")

# 使用try-except包装模型加载
try:
    model = Qwen2_5_VLWithSafetyClassifiers.from_pretrained(
        model_dir,
        torch_dtype=torch.float16 if device.startswith('cuda') else torch.float32,
        device_map=device
    )
    processor = AutoProcessor.from_pretrained(model_dir)
    print(f"模型加载完成，已移动到设备 {device}")
    model = model.eval()  # 设置为评估模式
except Exception as e:
    print(f"模型加载失败: {e}")
    print(traceback.format_exc())
    raise


def evaluate_safety_classifier_jsonl(model, processor, dataset_path):
    """评估安全分类器的性能 - 使用JSONL格式的数据集"""

    # 设置环境变量以帮助调试CUDA错误
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # 创建日志文件
    log_file = os.path.join(os.path.dirname(dataset_path), "safety_evaluation_log.txt")
    print(f"评估日志将保存到: {log_file}")

    # 初始化日志
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"安全分类器评估开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据集路径: {dataset_path}\n")
        f.write(f"模型路径: {model_dir}\n")
        f.write(f"使用设备: {device}\n\n")
        f.write("=" * 80 + "\n\n")

    # 定义四个分类任务的名称
    tasks = ["text_safety", "text_category", "image_safety", "image_category"]

    # 存储所有预测和真实标签
    all_true_labels = {task: [] for task in tasks}
    all_pred_labels = {task: [] for task in tasks}

    # 加载JSONL数据
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    print(f"加载了 {len(data)} 条数据进行评估")

    # 将基本信息写入日志
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"数据集大小: {len(data)} 条\n\n")

    # 记录成功处理的样本数
    successful_samples = 0
    batch_successful = 0  # 用于记录每批次成功处理的样本数
    batch_size = 200  # 每200个样本输出一次统计
    batch_start_time = time.time()  # 记录批次开始时间

    # 用于中间评估的临时变量
    batch_true_labels = {task: [] for task in tasks}
    batch_pred_labels = {task: [] for task in tasks}

    # 遍历数据集
    for idx, item in enumerate(tqdm(data, desc="评估中")):
        try:
            print(f"\n--------- 处理样本 {idx} ---------")

            # 获取标签
            try:
                raw_labels = item.get("label", None)
                if raw_labels is None:
                    print(f"警告: 样本 {idx} 没有标签，使用默认值")
                    labels = [0, 0, 0, 0]
                else:
                    if isinstance(raw_labels, list):
                        if len(raw_labels) != 4:
                            print(f"警告: 样本 {idx} 标签数量不正确 {raw_labels}, 使用默认值")
                            labels = [0, 0, 0, 0]
                        else:
                            labels = raw_labels
                    else:
                        print(f"警告: 样本 {idx} 标签格式不正确 {type(raw_labels)}, 使用默认值")
                        labels = [0, 0, 0, 0]

                # 验证标签值
                for i, (task, label) in enumerate(zip(tasks, labels)):
                    if task in ["text_safety", "image_safety"]:
                        if label not in [0, 1]:
                            print(f"警告: {task} 的标签值 {label} 不合法，设置为0")
                            labels[i] = 0
                    else:
                        if not isinstance(label, (int, np.integer)) or label < 0:
                            print(f"警告: {task} 的标签值 {label} 不合法，设置为0")
                            labels[i] = 0

                print(f"样本 {idx} 的处理后标签: {labels}")

                for i, task in enumerate(tasks):
                    all_true_labels[task].append(labels[i])

            except Exception as e:
                print(f"处理样本 {idx} 的标签时出错: {e}")
                print(traceback.format_exc())
                labels = [0, 0, 0, 0]
                for task in tasks:
                    all_true_labels[task].append(0)

            # 获取图像路径和文本
            image_path = item.get("images", [""])[0]
            print(f"样本 {idx} 的图像路径: {image_path}")

            # 获取消息内容
            messages = item.get("messages", [])
            text = ""
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    text = content.replace("<image>", "").strip()

            print(f"样本 {idx} 的文本: {text[:100]}..." if len(text) > 100 else f"样本 {idx} 的文本: {text}")

            # 如果图像路径为空或不存在，跳过此样本
            if not image_path or not os.path.exists(image_path):
                print(f"警告: 图像不存在 {image_path}, 跳过此样本")
                for task in tasks:
                    all_pred_labels[task].append(0)
                continue

            try:
                # 使用processor处理图像和文本
                print(f"样本 {idx} 开始处理...")

                # 读取图像
                image = Image.open(image_path).convert('RGB')

                # 构建消息格式
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image,
                                "min_pixels": 224 * 224,
                                "max_pixels": 1280 * 28 * 28,
                            },
                            {"type": "text", "text": text}
                        ]
                    }
                ]

                # 使用processor处理输入 - 按照官方推荐的方式
                try:
                    # 第1步：处理会话模板获取文本
                    processed_text = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )

                    # 第2步：处理视觉信息（图像和视频）
                    image_inputs, video_inputs = process_vision_info(messages)

                    # 第3步：将文本和视觉信息组合起来，进行处理
                    inputs = processor(
                        text=[processed_text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt"
                    )

                    # 打印输入形状以进行调试
                    print("输入形状:")
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            print(f"- {k}: {v.shape}")

                    # 移动到正确的设备
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    print(f"样本 {idx} 输入处理完成")
                except Exception as e:
                    print(f"处理输入时出错: {e}")
                    print(traceback.format_exc())
                    for task in tasks:
                        all_pred_labels[task].append(0)
                    continue

                # 模型推理
                with torch.no_grad():
                    try:
                        if device.startswith('cuda'):
                            torch.cuda.empty_cache()

                        outputs = model(**inputs)
                        print(f"样本 {idx} 模型推理完成")

                        # 获取安全预测结果
                        safety_preds = model.get_safety_predictions()

                        if safety_preds is None or not isinstance(safety_preds, dict):
                            print(f"警告: 样本 {idx} 的预测结果无效")
                            for task in tasks:
                                all_pred_labels[task].append(0)
                            continue

                        # 打印详细的预测结果
                        print(f"\n样本 {idx} 预测结果详情:")
                        print("=" * 50)
                        for task in tasks:
                            if task in safety_preds:
                                pred_value = safety_preds[task]
                                if hasattr(pred_value, 'cpu'):
                                    pred_value = pred_value.cpu().item()
                                print(f"{task}: {pred_value}")

                                # 如果是类别预测，显示具体类别名称
                                if task == "text_category" and pred_value < len(SAFETY_CATEGORIES):
                                    print(f"  类别名称: {SAFETY_CATEGORIES[pred_value]}")
                                elif task == "image_category" and pred_value < len(SAFETY_CATEGORIES):
                                    print(f"  类别名称: {SAFETY_CATEGORIES[pred_value]}")
                            else:
                                print(f"{task}: 无预测结果")
                        print("=" * 50)

                        # 比较预测结果与真实标签
                        true_labels_for_sample = labels
                        print(f"真实标签: {true_labels_for_sample}")

                        print("\n预测与真实标签比较:")
                        for i, task in enumerate(tasks):
                            true_val = true_labels_for_sample[i]
                            pred_val = safety_preds.get(task, None)
                            if pred_val is not None and hasattr(pred_val, 'cpu'):
                                pred_val = pred_val.cpu().item()

                            match = "✓" if pred_val == true_val else "✗"
                            print(f"{task}: 预测={pred_val}, 真实={true_val} {match}")

                            # 如果是类别预测且有效，显示类别名称
                            if task in ["text_category", "image_category"] and true_val < len(SAFETY_CATEGORIES) and isinstance(pred_val, (int, float)):
                                if int(pred_val) < len(SAFETY_CATEGORIES):
                                    print(f"  预测类别: {SAFETY_CATEGORIES[int(pred_val)]}")
                                if true_val < len(SAFETY_CATEGORIES):
                                    print(f"  真实类别: {SAFETY_CATEGORIES[true_val]}")
                        print("=" * 50)

                        # 处理预测结果
                        for task in tasks:
                            if task in safety_preds:
                                pred_value = safety_preds[task]
                                if pred_value is None:
                                    all_pred_labels[task].append(0)
                                    batch_pred_labels[task].append(0)  # 添加到批次预测标签
                                else:
                                    try:
                                        pred = int(pred_value.cpu().item() if hasattr(
                                            pred_value, 'cpu') else pred_value)
                                        pred = 1 if pred == 1 and task in [
                                            "text_safety", "image_safety"] else max(0, pred)
                                        all_pred_labels[task].append(pred)
                                        batch_pred_labels[task].append(pred)  # 添加到批次预测标签
                                    except Exception as e:
                                        print(f"警告: 处理任务 {task} 的预测值时出错: {e}")
                                        all_pred_labels[task].append(0)
                                        batch_pred_labels[task].append(0)  # 添加到批次预测标签
                            else:
                                all_pred_labels[task].append(0)
                                batch_pred_labels[task].append(0)  # 添加到批次预测标签

                        # 添加标签到批次真实标签
                        for i, task in enumerate(tasks):
                            batch_true_labels[task].append(labels[i])

                        successful_samples += 1
                        batch_successful += 1
                        print(f"样本 {idx} 处理成功!")

                        # 每处理batch_size个样本，计算一次中间结果
                        if (idx + 1) % batch_size == 0 or idx == len(data) - 1:
                            batch_end_time = time.time()
                            batch_time = batch_end_time - batch_start_time

                            # 计算当前批次的平均处理时间
                            avg_sample_time = batch_time / max(1, batch_successful)

                            batch_msg = f"\n============ 批次处理统计 [{idx-batch_size+1 if idx >= batch_size else 0}-{idx}] ============\n"
                            batch_msg += f"处理样本数: {batch_size}, 成功: {batch_successful}, 失败: {batch_size - batch_successful}\n"
                            batch_msg += f"批次处理时间: {batch_time:.2f}秒, 平均每样本: {avg_sample_time:.2f}秒\n"
                            batch_msg += f"总体进度: {idx+1}/{len(data)} ({(idx+1)/len(data)*100:.2f}%)\n"

                            # 计算当前批次的评估指标
                            if batch_successful > 0:
                                batch_results = {}
                                for task in tasks:
                                    # 确保有足够样本进行评估
                                    if len(batch_true_labels[task]) > 0 and len(batch_pred_labels[task]) > 0:
                                        batch_accuracy = accuracy_score(
                                            batch_true_labels[task], batch_pred_labels[task])

                                        if task in ["text_safety", "image_safety"]:
                                            batch_precision, batch_recall, batch_f1, _ = precision_recall_fscore_support(
                                                batch_true_labels[task], batch_pred_labels[task], average='binary', zero_division=0
                                            )
                                            batch_results[task] = {
                                                "accuracy": batch_accuracy,
                                                "precision": batch_precision,
                                                "recall": batch_recall,
                                                "f1": batch_f1
                                            }
                                        else:
                                            batch_precision, batch_recall, batch_f1, _ = precision_recall_fscore_support(
                                                batch_true_labels[task], batch_pred_labels[task], average='macro', zero_division=0
                                            )
                                            batch_results[task] = {
                                                "accuracy": batch_accuracy,
                                                "macro_precision": batch_precision,
                                                "macro_recall": batch_recall,
                                                "macro_f1": batch_f1
                                            }

                                # 打印批次评估结果
                                batch_msg += "\n批次评估结果:\n"
                                for task, metrics in batch_results.items():
                                    batch_msg += f"{task}:\n"
                                    for metric_name, metric_value in metrics.items():
                                        batch_msg += f"  - {metric_name}: {metric_value:.4f}\n"

                            print(batch_msg)

                            # 写入日志
                            with open(log_file, "a", encoding="utf-8") as f:
                                f.write(batch_msg + "\n")

                            # 重置批次计数器和标签列表
                            batch_successful = 0
                            batch_start_time = time.time()
                            batch_true_labels = {task: [] for task in tasks}
                            batch_pred_labels = {task: [] for task in tasks}

                    except RuntimeError as e:
                        print(f"运行时错误: {e}")
                        print(traceback.format_exc())
                        for task in tasks:
                            all_pred_labels[task].append(0)
                        continue

            except Exception as e:
                print(f"处理样本 {idx} 时出错: {e}")
                print(traceback.format_exc())
                for task in tasks:
                    all_pred_labels[task].append(0)
                continue

        except Exception as e:
            print(f"处理样本 {idx} 时出错: {e}")
            print(traceback.format_exc())
            for task in tasks:
                all_pred_labels[task].append(0)

    print(f"成功处理 {successful_samples}/{len(data)} 个样本")

    # 将最终统计信息写入日志
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n\n================ 最终评估结果 ================\n")
        f.write(f"总样本数: {len(data)}, 成功处理: {successful_samples}, 成功率: {successful_samples/len(data)*100:.2f}%\n\n")

    # 确保所有标签列表长度一致
    min_length = min(len(all_true_labels[task]) for task in tasks)
    min_length = min(min_length, min(len(all_pred_labels[task]) for task in tasks))

    for task in tasks:
        all_true_labels[task] = all_true_labels[task][:min_length]
        all_pred_labels[task] = all_pred_labels[task][:min_length]

    # 如果没有有效样本，返回空结果
    if min_length == 0:
        print("警告: 没有成功处理任何样本，无法计算评估指标")
        return {}

    # 评估每个任务的性能
    results = {}
    for task in tasks:
        if len(all_true_labels[task]) == 0 or len(all_pred_labels[task]) == 0:
            print(f"警告: {task} 没有有效的预测数据")
            continue

        true = all_true_labels[task]
        pred = all_pred_labels[task]

        print(f"\n{task} 有效样本数: {len(true)}")

        # 计算指标
        accuracy = accuracy_score(true, pred)
        cm = confusion_matrix(true, pred)

        if task in ["text_safety", "image_safety"]:
            precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average='binary', zero_division=0)
            results[task] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": cm.tolist()
            }
        else:
            precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average='macro', zero_division=0)
            results[task] = {
                "accuracy": accuracy,
                "macro_precision": precision,
                "macro_recall": recall,
                "macro_f1": f1,
                "confusion_matrix": cm.tolist()
            }

        # 打印评估结果
        print(f"\n{task} 评估结果:")
        print(f"准确率: {accuracy:.4f}")
        if task in ["text_safety", "image_safety"]:
            print(f"精确率: {precision:.4f}")
            print(f"召回率: {recall:.4f}")
            print(f"F1分数: {f1:.4f}")
        else:
            print(f"宏平均精确率: {precision:.4f}")
            print(f"宏平均召回率: {recall:.4f}")
            print(f"宏平均F1分数: {f1:.4f}")
        print(f"混淆矩阵:\n{cm}")

        # 将结果写入日志
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{task} 评估结果:\n")
            f.write(f"准确率: {accuracy:.4f}\n")
            if task in ["text_safety", "image_safety"]:
                f.write(f"精确率: {precision:.4f}\n")
                f.write(f"召回率: {recall:.4f}\n")
                f.write(f"F1分数: {f1:.4f}\n")
            else:
                f.write(f"宏平均精确率: {precision:.4f}\n")
                f.write(f"宏平均召回率: {recall:.4f}\n")
                f.write(f"宏平均F1分数: {f1:.4f}\n")
            f.write(f"混淆矩阵:\n{cm}\n\n")

    # 写入JSON结果
    json_result_path = os.path.join(os.path.dirname(dataset_path), "safety_evaluation_results.json")
    with open(json_result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n评估完成，详细日志保存在: {log_file}")
    print(f"评估结果JSON保存在: {json_result_path}")

    return results


# 调用安全分类器评估函数
if __name__ == "__main__":
    dataset_path = "/home/hxm826/ms-swift/examples/custom/mm-safety-ds/mm_safety_dataset.jsonl"
    evaluate_safety_classifier_jsonl(model, processor, dataset_path)
