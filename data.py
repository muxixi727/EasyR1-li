import os
import random
import json
import io
from PIL import Image as I
from datasets import Dataset, Features, Sequence, Value, Image
import pandas as pd
from tqdm import tqdm

# 假设你有一个函数来检查图像的大小
def has_valid_image_size_from_path(image_path):
    try:
        with I.open(image_path) as img:
            return img.size[0] > 100 and img.size[1] > 100  # 例如检查图像是否大于100x100
    except:
        return False

def image_to_bytes(image):
    """将PIL图像对象转换为字节流"""
    byte_io = io.BytesIO()
    image.save(byte_io, format="PNG")  # 可以改成你需要的格式，如 PNG、JPEG
    byte_io.seek(0)
    return byte_io.read()

def process_data(input_file):
    with open(input_file, 'r') as f:
        data_all = json.load(f)

    data_all_filtered = []
    for data_tmp in tqdm(data_all, desc="Processing images", unit="image"):
        image_path = data_tmp['image']
        if has_valid_image_size_from_path(image_path):
            data_all_filtered.append(data_tmp)
        else:
            print(f"Invalid image: {image_path}")

    print(f'len(data_all): {len(data_all)}')
    print(f'len(data_all_filtered): {len(data_all_filtered)}')

    random.shuffle(data_all_filtered)

    # 使用生成器直接生成数据
    def generate_data():
        for item in tqdm(data_all_filtered, desc="Generating dataset", unit="example"):
            image_path = item.get('image')
            problem = "<image>" + item['problem']
            solution = item['answer']

            # img = I.open(image_path)

            # 将图像转换为字节流
            # image_bytes = image_to_bytes(img)

            # 使用字节流直接构建数据
            yield {
                # 'images': [{'bytes': image_bytes, 'path': image_path}],  # 将字节流包装成列表
                'images': [image_path],
                'problem': problem,
                'answer': solution
            }

    # 使用 Dataset.from_generator 创建数据集
    features = Features({
        # 'images': Sequence({'bytes': Value('binary'), 'path': Value('string')}),  # 存储字节流的序列
        'images': Sequence(Value('string')),
        'problem': Value('string'),
        'answer': Value('string')
    })

    # train_dataset = Dataset.from_generator(generate_data, features=features)
    train_dataset = Dataset.from_generator(generate_data)
    # trainset = train_dataset.cast_column("images", Sequence(Image()))

    # 打印数据集信息
    print(train_dataset)
    train_save_path = "train-00000-of-00001.parquet"
    test_save_path = "test-00000-of-00001.parquet"
    if "train" in input_file:
        os.makedirs("/lpai/output/train", exist_ok=True)
        train_dataset.to_parquet(os.path.join("/lpai/output/train", train_save_path))
    else:
        os.makedirs("/lpai/output/test", exist_ok=True)
        train_dataset.to_parquet(os.path.join("/lpai/output/test", test_save_path))
    
    print(f"Saved")

if __name__ == "__main__":
    # input_files = ['/lpai/EasyR1/data/train_data.json']
    input_files = ['/lpai/dataset/anomaly-detection-parquet/0-1-0/test_data.json', '/lpai/dataset/anomaly-detection-parquet/0-1-0/train_data.json']
    for file in input_files:# 填入你的 JSON 文件路径列表
        process_data(file)
