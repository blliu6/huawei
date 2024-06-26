import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pickle

# 定义模型
class Model(nn.Module):
    def __init__(self, dense, hidden_dim):
        super(Model, self).__init__()
        self.seq = nn.Sequential()
        s = 30720//2
        for i in range(dense):
            self.seq.add_module(f'layer_{i}', nn.Linear(s, hidden_dim))
            self.seq.add_module(f'bn_{i}',nn.BatchNorm1d(hidden_dim))
            self.seq.add_module(f'activate_{i}', nn.PReLU(hidden_dim))
            self.seq.add_module(f'drop_{i}',nn.Dropout(0.1))
            s = hidden_dim
        self.seq.add_module(f'layer_last', nn.Linear(s, 11))

    def forward(self, x):
        x = x
        return self.seq(x)

def get_data(data):
    l = 1
    r = 15360 // l
    a = data[::2]
    b = data[1::2]
    p = [complex(x, y) for x, y in zip(a,b)]
    result = []
    for i in range(l):
        q = p[i*r:(i+1)*r]
        q_ = np.fft.ifft(q)
        q_ = np.array([e.real for e in q_], dtype=np.float32)
        result.append(q_)
    return np.concatenate(result)

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
                # data.append(get_data(data_row_float16))
    return data, labels


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


def collate_fn(batch):
    features = []
    labels = []
    for _, item in enumerate(batch):
        features.append(item['data'])
        labels.append(item['label'])
    return torch.stack(features, 0), torch.stack(labels, 0)

def plot(item):
    plt.plot(item)
    plt.savefig('a.jpg')    
    # plt.clf()
    # plt.plot(item[1:2000:2])
    # plt.savefig('b.jpg') 

if __name__ == '__main__':
    torch.manual_seed(2024)

    # 加载数据
    # folder_path = "/home/blliu/huawei/train_set_remake"
    # data, labels = preprocess(folder_path)
    # plot(data[1])
    # torch.save({
    #         'data': data,
    #         'labels': labels,
    #     }, 'a.pth')


    r = torch.load('a.pth')
    data = r['data']    
    labels = r['labels']
    # 划分数据集
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
    # train_data, train_labels = data, labels
    # val_data, val_labels = data.copy(), labels.copy()
    train_dataset = ComplexDataset(train_data, train_labels)
    val_dataset = ComplexDataset(val_data, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)

    # 定义模型
    model = Model(2, 1024)
    criterion = nn.CrossEntropyLoss()

    # optimizer = optim.RMSprop(model.parameters(), lr=0.001, weight_decay=1e-5)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

    # 训练模型
    num_epochs = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    ans = 0
    for epoch in range(num_epochs):
        model.train()
        loss_all = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # outputs = torch.softmax(outputs, 1)
            loss = criterion(outputs, targets)
            loss.backward()
            loss_all += loss.item()
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
        print(f'Epoch {epoch + 1}, loss:{loss_all}, Validation Accuracy: {accuracy}')
        if accuracy > ans:
            torch.save(model.state_dict(), f'./model/model.pth')
            ans = accuracy
    
    with open('./model/ans.txt', 'w') as f:
        f.write(str(ans))
    print(ans)
    # torch.save(model, f'./model/model_{int(accuracy * 10000)}.pth')
    # torch.save(model.state_dict(), f'./model/model.pth')
