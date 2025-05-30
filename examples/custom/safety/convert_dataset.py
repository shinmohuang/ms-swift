#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import argparse
from pathlib import Path


def convert_to_swift_format(input_file, output_file):
    """
    将原始数据集转换为Swift多模态序列分类格式

    Args:
        input_file (str): 输入文件路径
        output_file (str): 输出文件路径
    """
    print(f"读取源文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"源数据集包含 {len(data)} 条记录")

    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 定义安全评级映射
    safety_rating_map = {
        "Safe": 0,
        "Unsafe": 1
    }

    # 创建固定的类别映射字典
    text_category_map = {
        "NA: None to Apply": 0,
        "harassment": 1,
        "harassment_threatening": 2,
        "hate": 3,
        "hate_threatening": 4,
        "illicit": 5,
        "illicit_violent": 6,
        "self_harm": 7,
        "self_harm_instructions": 8,
        "self_harm_intent": 9,
        "sexual": 10,
        "sexual_minors": 11,
        "violence": 12,
        "violence_graphic": 13
    }

    print(f"类别映射: {text_category_map}")

    img_category_map = {
        "NA: None to Apply": 0,
        "self_harm": 1,
        "self_harm_instructions": 2,
        "self_harm_intent": 3,
        "sexual": 4,
        "violence": 5,
        "violence_graphic": 6
    }

    print(f"类别映射: {img_category_map}")

    # 转换为Swift格式
    converted_data = []
    for item in data:
        # 获取安全评级的数字编码
        text_safety = safety_rating_map.get(item["text_safety_rating"], 0)
        img_safety = safety_rating_map.get(item["img_safety_rating"], 0)

        # 通过category_scores找出分数最高的类别
        text_category = item["text_category"]
        img_category = item["img_category"]

        # 如果存在category_scores，则使用分数最高的类别
        if "text_category_scores" in item and item["text_category_scores"]:
            # 找出分数最高的类别
            max_score = -1
            max_category = "NA: None to Apply"
            for cat, score in item["text_category_scores"].items():
                # 处理score可能为None的情况
                if score is not None and score > max_score:
                    max_score = score
                    max_category = cat

            # 如果找到了分数较高的类别，则使用该类别
            if max_score > 0:
                text_category = max_category

        if "img_category_scores" in item and item["img_category_scores"]:
            # 找出分数最高的类别
            max_score = -1
            max_category = "NA: None to Apply"
            for cat, score in item["img_category_scores"].items():
                # 处理score可能为None的情况
                if score is not None and score > max_score:
                    max_score = score
                    max_category = cat

            # 如果找到了分数较高的类别，则使用该类别
            if max_score > 0:
                img_category = max_category

        # 获取类别的数字编码
        text_cat_code = text_category_map.get(text_category, 0)
        img_cat_code = img_category_map.get(img_category, 0)

        # 创建Swift格式的数据项
        swift_item = {
            "messages": [
                {
                    "role": "user",
                    "content": f"<image>{item['text']}"
                }
            ],
            "images": item["images"],
            "id": item["id"],
            "label": [
                text_safety,
                text_cat_code,
                img_safety,
                img_cat_code
            ],
            "similarity_score": item["similarity_score"]
        }

        # 处理images字段 - 确保它是列表格式
        if isinstance(swift_item["images"], str):
            swift_item["images"] = [swift_item["images"]]

        converted_data.append(swift_item)

    # 将转换后的数据写入JSONL格式
    print(f"写入转换后的数据到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 保存类别映射信息，方便后续使用
    mapping_file = os.path.join(os.path.dirname(output_file), "category_mapping.json")
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump({
            "safety_rating_map": safety_rating_map,
            "text_category_map": text_category_map,
            "img_category_map": img_category_map
        }, f, indent=2, ensure_ascii=False)
    print(f"类别映射信息已保存到: {mapping_file}")

    print(f"转换完成! 共转换 {len(converted_data)} 条记录")


def main():
    parser = argparse.ArgumentParser(description='将数据集转换为Swift多模态序列分类格式')
    parser.add_argument('--input', type=str,
                        default="examples/custom/combined_similarity_results_with_moderation.json", help='输入JSON文件路径')
    parser.add_argument('--output', type=str,
                        default="examples/custom/mm-safety-ds/mm_safety_dataset_with_similarity.jsonl", help='输出JSONL文件路径')

    args = parser.parse_args()
    convert_to_swift_format(args.input, args.output)


if __name__ == '__main__':
    main()
