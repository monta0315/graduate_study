import tensorflow as tf
import os
import time
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import numpy as np

#tf2.0なのに@tf.functionを使用していないので死ぬほど時間かかります

#データセッティング
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


#データ加工
x_train = x_train.astype(np.float32).reshape(x_train.shape[0], 784) / 255.0
x_test = x_test.astype(np.float32).reshape(x_test.shape[0], 784) / 255.0


#modelを定義
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(784,)),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Activation('softmax')
])

model.summary()
#各種パラメータを宣言
loss = tf.keras.losses.SparseCategoricalCrossentropy()
#モデルをコンパイルする
model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])


#モデルを学習させる
model.fit(x_train, y_train,epochs=5)

#推論時間の測定
n_loop = 5
start = time.perf_counter()
for n in range(n_loop):
  for x in x_test:
    model(np.array([x]))

print('elapsed time for {} prediction {} [msec]'.format(
    len(x_test), (time.perf_counter()-start) * 1000 / n_loop))
