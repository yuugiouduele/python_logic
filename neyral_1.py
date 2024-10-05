 import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        # 他の畳み込み層やプーリング層を追加

    def forward(self, x):
        x = self.conv1(x)
        # 他の層の処理を追加
        return x

# モデルのインスタンスを作成
model = CNN()
print(model)
