import os
import re
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from train import Model


def preprocess_test(folder_path):
    data = {}
    cnt = 0
    for filename in os.listdir(folder_path):
        cnt += 1
        match = re.search(r'(\d+).bin', filename)
        if match:
            label = int(match.group(1))
        else:
            continue
        if filename.endswith(".bin"):
            with open(os.path.join(folder_path, filename), 'rb') as file:
                data_row_bin = file.read()
                data_row_float16 = np.frombuffer(data_row_bin, dtype=np.float16)  # 原始数据是float16，直接把二进制bin读成float16的数组
                data_row_float16 = np.array(data_row_float16)
                data[label] = data_row_float16
    data = dict(sorted(data.items()))
    data = list(data.values())
    return data


class ComplexDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'data': torch.tensor(self.data[idx], dtype=torch.float32)}
        return sample


def collate_fn(batch):
    features = []
    for _, item in enumerate(batch):
        features.append(item['data'])
    return torch.stack(features, 0)


if __name__ == '__main__':
    folder_path = "/home/blliu/huawei/test_set"
    data = preprocess_test(folder_path)

    # 创建测试数据集实例
    test_dataset = ComplexDataset(data)
    # 构建test_loader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # 加载模型
    model_path = './model/model.pth'  # 替换为你的模型路径

    model = Model(2, 1024)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 假设test_loader用于加载无标签的测试数据
    predictions = []

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # 获取最大概率的类别索引作为预测结果
            predictions.extend(preds.cpu().numpy())  # 将预测结果从GPU转移到CPU，并添加到列表中

    df_predictions = pd.DataFrame({'Prediction': predictions})

    # 将预测结果保存到CSV文件，提交时注意去除表头
    csv_output_path = '../result.csv'
    df_predictions.to_csv(csv_output_path, index=False, header=False)  # index=False避免将索引写入CSV文件

    print(f'Predictions have been saved to {csv_output_path}')
