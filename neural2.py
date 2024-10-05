 import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()  # 入力画像を平滑化
        self.fc1 = nn.Linear(28*28, 128)  # 入力層から隠れ層への全結合層
        self.relu = nn.ReLU()  # 活性化関数（ReLU）
        self.fc2 = nn.Linear(128, 10)  # 隠れ層から出力層への全結合層

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# モデルのインスタンスを作成
model = SimpleNN()
print(model)
