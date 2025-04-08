#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import argparse
from collections import Counter


def validate_swift_format(input_file, mapping_file=None):
    """
    验证数据集是否符合Swift的多模态序列分类格式

    Args:
        input_file (str): 输入JSONL文件路径
        mapping_file (str, optional): 类别映射文件路径
    """
    print(f"验证数据集文件: {input_file}")

    # 加载类别映射（如果提供）
    category_mapping = None
    if mapping_file and os.path.exists(mapping_file):
        with open(mapping_file, 'r', encoding='utf-8') as f:
            category_mapping = json.load(f)
            print(f"加载类别映射: {mapping_file}")

    # 统计信息
    total_items = 0
    valid_items = 0
    invalid_items = 0
    missing_fields = Counter()
    label_stats = {
        "text_safety": Counter(),
        "text_category": Counter(),
        "img_safety": Counter(),
        "img_category": Counter()
    }

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                total_items += 1

                # 验证必需字段
                required_fields = ["messages", "images", "label"]
                is_valid = True

                for field in required_fields:
                    if field not in item:
                        is_valid = False
                        missing_fields[field] += 1

                # 验证messages结构
                if "messages" in item:
                    if not isinstance(item["messages"], list) or len(item["messages"]) == 0:
                        is_valid = False
                        missing_fields["messages_invalid"] += 1
                    else:
                        message = item["messages"][0]
                        if "role" not in message or "content" not in message:
                            is_valid = False
                            missing_fields["message_fields"] += 1
                        elif not message["content"].startswith("<image>"):
                            missing_fields["no_image_tag"] += 1

                # 验证images字段
                if "images" in item:
                    if not isinstance(item["images"], list) or len(item["images"]) == 0:
                        is_valid = False
                        missing_fields["images_invalid"] += 1

                # 验证label字段
                if "label" in item:
                    if not isinstance(item["label"], list) or len(item["label"]) != 4:
                        is_valid = False
                        missing_fields["label_invalid"] += 1
                    else:
                        # 统计标签分布
                        label_stats["text_safety"][item["label"][0]] += 1
                        label_stats["text_category"][item["label"][1]] += 1
                        label_stats["img_safety"][item["label"][2]] += 1
                        label_stats["img_category"][item["label"][3]] += 1

                if is_valid:
                    valid_items += 1
                else:
                    invalid_items += 1

            except json.JSONDecodeError:
                print(f"错误: 第{line_num}行无法解析为JSON")
                invalid_items += 1

    # 打印统计信息
    print(f"\n====== 验证完成 ======")
    print(f"总记录数: {total_items}")
    print(f"有效记录: {valid_items} ({valid_items/total_items*100:.2f}%)")
    print(f"无效记录: {invalid_items} ({invalid_items/total_items*100:.2f}%)")

    if missing_fields:
        print("\n缺失或无效字段统计:")
        for field, count in missing_fields.items():
            print(f"  - {field}: {count}")

    print("\n标签分布统计:")

    # 反向映射函数（将数字转回类别名称）
    def get_reverse_mapping(mapping_dict, key_type):
        if not category_mapping or key_type not in category_mapping:
            return {}
        return {v: k for k, v in category_mapping[key_type].items()}

    # 文本安全评级
    reverse_safety = get_reverse_mapping(category_mapping, "safety_rating_map")
    print("\n文本安全评级 (text_safety):")
    for value, count in label_stats["text_safety"].most_common():
        label_name = reverse_safety.get(value, f"未知类别({value})")
        print(f"  - {value} ({label_name}): {count} ({count/total_items*100:.2f}%)")

    # 文本类别
    reverse_category = get_reverse_mapping(category_mapping, "category_map")
    print("\n文本类别 (text_category):")
    for value, count in label_stats["text_category"].most_common(20):
        label_name = reverse_category.get(value, f"未知类别({value})")
        print(f"  - {value} ({label_name}): {count} ({count/total_items*100:.2f}%)")

    # 图像安全评级
    print("\n图像安全评级 (img_safety):")
    for value, count in label_stats["img_safety"].most_common():
        label_name = reverse_safety.get(value, f"未知类别({value})")
        print(f"  - {value} ({label_name}): {count} ({count/total_items*100:.2f}%)")

    # 图像类别
    print("\n图像类别 (img_category):")
    for value, count in label_stats["img_category"].most_common(20):
        label_name = reverse_category.get(value, f"未知类别({value})")
        print(f"  - {value} ({label_name}): {count} ({count/total_items*100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description='验证Swift多模态序列分类数据集格式')
    parser.add_argument('--input', type=str, required=True, help='输入JSONL文件路径')
    parser.add_argument('--mapping', type=str, help='类别映射JSON文件路径')

    args = parser.parse_args()
    validate_swift_format(args.input, args.mapping)


if __name__ == '__main__':
    main()
