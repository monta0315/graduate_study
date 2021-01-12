import tensorflow as tf
import os
import time
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import tensorflow.keras.layers as layers
import numpy as np

#tf2.0なのに@tf.functionを使用していないので死ぬほど時間かかります

#データセッティング
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#データ加工
x_train = x_train.astype(np.float32).reshape(x_train.shape[0], 784) / 255.0
x_test = x_test.astype(np.float32).reshape(x_test.shape[0], 784) / 255.0

""" def create_model():
  inputs = layers.Input((imageDim,))
  x = layers.Dense(10)(inputs)
  x = layers.Softmax()(x)
  return tf.keras.models.Model(inputs, x)
 """

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(784,)))
model.add(tf.keras.layers.Dense(10,activation='softmax'))

model.summary()
#各種パラメータを宣言
loss = tf.keras.losses.SparseCategoricalCrossentropy()
acc = tf.keras.metrics.SparseCategoricalAccuracy()
optim = tf.keras.optimizers.Adam()

#モデルをコンパイルする
model.compile(optimizer=optim, loss=loss, metrics=[acc])


#モデルを学習させる
model.fit(x_train, y_train,epochs=5)



@tf.function
def test_step(x):
  model(x)

n_loop = 1
start = time.perf_counter()
for n in range(n_loop):
  for x in x_test:
    test_step(x)

print('elapsed time for {} prediction {} [msec]'.format(
    len(x_test), (time.perf_counter()-start) * 1000 / n_loop))
