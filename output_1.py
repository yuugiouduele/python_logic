import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense

# 語彙数と埋め込み次元数を指定
vocab_size = 10000
embedding_dim = 128

# 出力層の作成
model = tf.keras.Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim),
    LSTM(64),
    Dense(vocab_size, activation='softmax')  # 単語ごとの確率を出力
])

# モデルのコンパイル
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
