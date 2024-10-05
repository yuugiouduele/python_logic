 import tensorflow as tf
from tensorflow.keras.layers import Dense

# クラス数（カテゴリ数）を指定
num_classes = 10

# 出力層の作成
model = tf.keras.Sequential([
    # 他の層を追加
    Dense(num_classes, activation='softmax')  # クラスごとの確率を出力
])

# モデルのコンパイル
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
