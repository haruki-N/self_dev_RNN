import numpy as np


# 1 step の処理を行う RNN クラスの実装
class RNN:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next)   # 逆伝搬のためにキャッシュしておく

        return h_next
    
    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dt = dh_next * (1 - h_next**2)  # tanhの逆伝搬
        db = np.sum(dt, axis=0)
        dWh = np.dot(h_prev.T, dt)   # 行列積の逆伝搬
        dh_prev = np.dot(dt, Wh.T)   # 行列積の逆伝搬
        dWx = np.dot(x.T, dt)        # 行列積の逆伝搬
        dx = np.dot(dt, Wx.T)        # 行列積の逆伝搬

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev



# T step の処理をまとめて行う TimeRNN クラスの実装
class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]

        self.layers = None
        self.h, self.dh = None, None
        self.stateful = stateful   # hidden state h を保持するかどうかを決めるフラグ: False で最初のRNNレイヤの隠れ状態を0にする


    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None

    def forward(self, xs):   # 入力 xs は T step 分のデータをまとめて受け取る
        Wx, Wh, b = self.params
        N, T, D = xs.shape  # バッチサイズ, 時系列データの数, 入力ベクトルの次元数
        D, H = Wx.shape     # 入力ベクトルの次元数, 隠れ状態ベクトルの次元数

        self.rnn_layers = []
        hs = np.empty((N, T, H), dtype='f')  # 隠れ状態ベクトルを保持する配列 (T step分)

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')  # 隠れ状態ベクトルを0で初期化

        for t in range(T):
            rnn_layer = RNN(*self.params)   # RNNレイヤの生成
            self.h = rnn_layer.forward(xs[:, t, :], self.h)  # RNNレイヤの順伝搬
            hs[:, t, :] = self.h
            self.rnn_layers.append(rnn_layer)   # RNNレイヤをリストに追加

        return hs


    def backward(self, dhs):   # 入力 dhs は T step 分の勾配をまとめて受け取る
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        # 初期化
        dxs = np.empty((N, T, D), dtype="f")
        dh = 0
        grads = [0, 0, 0]

        for t in reversed(range(T)):
            rnn_layer = self.rnn_layers[t]
            dx, dh = rnn_layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            for i, grad in enumerate(rnn_layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh

        return dxs