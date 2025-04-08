import os
import json
from PIL import Image
import tqdm
import argparse


def resize_image(image_path, output_path, max_size=512):
    """
    调整图片大小，保持宽高比，最大边长为max_size
    """
    try:
        with Image.open(image_path) as img:
            # 计算缩放比例
            width, height = img.size
            scale = min(max_size / width, max_size / height)

            # 如果图片已经小于目标大小，不需要缩放
            if scale >= 1:
                img.save(output_path)
                return True

            # 计算新尺寸并调整大小
            new_width, new_height = int(width * scale), int(height * scale)
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            resized_img.save(output_path)
            return True
    except Exception as e:
        print(f"处理图片 {image_path} 失败: {e}")
        return False


def process_dataset(dataset_path, output_dir, processed_image_dir, max_size=512):
    """
    处理JSONL格式的数据集，缩小所有图片
    """
    os.makedirs(processed_image_dir, exist_ok=True)
    output_jsonl = os.path.join(output_dir, "processed_dataset.jsonl")

    # 读取原始数据集
    with open(dataset_path, 'r') as f:
        lines = f.readlines()

    processed_lines = []
    success_count = 0
    fail_count = 0

    for line in tqdm.tqdm(lines, desc="处理图片"):
        try:
            data = json.loads(line.strip())
            if "images" in data and data["images"]:
                processed_images = []
                for image_path in data["images"]:
                    # 创建新的图片路径
                    image_filename = os.path.basename(image_path)
                    processed_image_path = os.path.join(processed_image_dir, image_filename)

                    # 调整图片大小
                    if resize_image(image_path, processed_image_path, max_size):
                        processed_images.append(processed_image_path)
                        success_count += 1
                    else:
                        # 如果处理失败，使用原始图片
                        processed_images.append(image_path)
                        fail_count += 1

                # 更新数据中的图片路径
                data["images"] = processed_images

            # 将处理后的数据写入新文件
            processed_lines.append(json.dumps(data, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"处理数据行失败: {e}")
            # 保留原始数据行
            processed_lines.append(line)

    # 写入处理后的数据集
    with open(output_jsonl, 'w') as f:
        f.writelines(processed_lines)

    print(f"成功处理 {success_count} 张图片，失败 {fail_count} 张图片")
    print(f"处理后的数据集保存在: {output_jsonl}")

    return output_jsonl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预处理多模态安全数据集中的图片")
    parser.add_argument("--dataset", type=str, required=True, help="原始数据集路径（JSONL格式）")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--max_size", type=int, default=512, help="图片最大边长")
    args = parser.parse_args()

    # 创建处理后图片的存放目录
    processed_image_dir = os.path.join(args.output_dir, "processed_images")

    # 处理数据集
    processed_dataset = process_dataset(
        args.dataset,
        args.output_dir,
        processed_image_dir,
        args.max_size
    )
