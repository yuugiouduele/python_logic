import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense

# MNISTデータセットの読み込み
(X_train, _), (X_test, _) = mnist.load_data()

# データの前処理
X_train = X_train.reshape(X_train.shape[0], 784) / 255
X_test = X_test.reshape(X_test.shape[0], 784) / 255

# オートエンコーダのモデル構築
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(784,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(784, activation='sigmoid'))

# モデルのコンパイル
model.compile(optimizer='adam', loss='binary_crossentropy')

# モデルの学習
model.fit(X_train, X_train, epochs=10, batch_size=256, shuffle=True, validation_data=(X_test, X_test))

# テストデータで再現画像を生成
decoded_images = model.predict(X_test)

# 画像の表示（オリジナルと再現画像）
import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_images[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
