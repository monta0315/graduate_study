import tensorflow as tf
import os
import time
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import tensorflow.keras.layers as layers
import numpy as np

#params
imageDim = 784
outputDim=10

#データセッティング
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#データ加工
x_train = x_train.astype(np.float32).reshape(x_train.shape[0], 784) / 255.0
x_test = x_test.astype(np.float32).reshape(x_test.shape[0], 784) / 255.0

#モデル作成関数、一応、predictとtrainingの際にデータを渡しても問題ないっぽい？？
def create_model():
  inputs = layers.Input((imageDim,))
  x = layers.Dense(10)(inputs)
  x=layers.Softmax()(x)
  return tf.keras.models.Model(inputs,x)

model = create_model()
model.summary()
#各種パラメータを宣言
loss = tf.keras.losses.SparseCategoricalCrossentropy()
acc = tf.keras.metrics.SparseCategoricalAccuracy()
optim = tf.keras.optimizers.Adam()

#モデルをコンパイルする
model.compile(optimizer=optim, loss=loss, metrics=[acc])

model.fit(x_train,y_train, validation_data=(x_test,y_test),epochs=1,batch_size=128)

#テストデータ作成
test_ds = tf.data.Dataset.from_tensor_slices(x_test)
#test_ds = test_ds.batch(128)


n_loop = 5
start=time.perf_counter()
for n in range(n_loop):
  for x in x_test:
    model(np.array([x]))

print('elapsed time for {} prediction {} [msec]'.format(
    len(x_test), (time.perf_counter()-start) * 1000 / n_loop))

model_int = model.get_concrete_function(

)
