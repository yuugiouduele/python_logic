import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Irisデータセットを読み込む
iris = datasets.load_iris()

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# ニューラルネットワークモデルを作成
model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)

# モデルを訓練
model.fit(X_train, y_train)

# テストデータで予測
y_pred = model.predict(X_test)

# 正解率を表示
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2f}")
