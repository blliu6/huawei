import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# 定义模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(30720, 128)
        self.fc2 = nn.Linear(128, 11)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def preprocess(folder_path):
    labels = []
    data = []
    cnt = 0
    for filename in os.listdir(folder_path):
        cnt += 1
        if filename.endswith(".bin"):
            match = re.search(r'label_(\d+)_', filename)
            if match:
                label = int(match.group(1))
            else:
                continue
            with open(os.path.join(folder_path, filename), 'rb') as file:
                data_row_bin = file.read()
                labels.append(label)
                data_row_float16 = np.frombuffer(data_row_bin, dtype=np.float16)  # 原始数据是float16，直接把二进制bin读成float16的数组
                data_row_float16 = np.array(data_row_float16)
                data.append(data_row_float16)
    return data, labels


# 加载数据
folder_path = "./trainset"
data, labels = preprocess(folder_path)

# 划分数据集
train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.4, random_state=42)


class ComplexDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'data': torch.tensor(self.data[idx], dtype=torch.float32),
                  'label': torch.tensor(self.labels[idx], dtype=torch.long)}
        return sample


train_dataset = ComplexDataset(train_data, train_labels)
val_dataset = ComplexDataset(val_data, val_labels)


def collate_fn(batch):
    features = []
    labels = []
    for _, item in enumerate(batch):
        features.append(item['data'])
        labels.append(item['label'])
    return torch.stack(features, 0), torch.stack(labels, 0)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# 定义模型
model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

# 训练模型
num_epochs = 10000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = torch.softmax(outputs, 1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # 验证集上的评估
    model.eval()
    with torch.no_grad():
        total_correct = 0
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == targets).sum().item()

    accuracy = total_correct / len(val_dataset)
    print(f'Epoch {epoch + 1}, Validation Accuracy: {accuracy}')
    torch.save(model, f'./model/model_{int(accuracy * 10000)}.pth')
