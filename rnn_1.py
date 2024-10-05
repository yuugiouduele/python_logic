import numpy as np

# シンプルなRNNの実装
def update_state(xk, sk, wx, wRec):
    return xk * wx + sk * wRec

def forward_states(X, wx, wRec):
    S = np.zeros((X.shape[0], X.shape[1] + 1))
    for k in range(X.shape[1]):
        S[:, k + 1] = update_state(X[:, k], S[:, k], wx, wRec)
    return S

# 入力データ
X = np.array([[0., 1., 0., 0., 0., 0., 0., 0., 0., 1.]])

# 重み係数
wx, wRec = 1.0, 1.0

# RNNの状態を計算
S = forward_states(X, wx, wRec)

# 出力
print("RNNの出力:", S[:, -1])
