import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np
import time

imageDim = 784


# MNISTデータセットを使用
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


#x_testをpreidctにそのまま投げたら死ぬほど時間かかったからreshapeしている
x_train = x_train.reshape(x_train.shape[0], imageDim)
x_test = x_test.reshape(x_test.shape[0], imageDim)

# 0~1へ正規化する
x_train, x_test = x_train / 255., x_test / 255.
#print(x_train.shape, x_test.shape, t_train.shape, t_test.shape)

#@tf.function内の関数にデータを渡す際に加工しなければならなかったため。batch(1)はつけないとエラー出る
test_ds = tf.data.Dataset.from_tensor_slices((x_test)).batch(1)

#test_dsの型を確認数
tf.print(test_ds)

#modelを定義
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(784,)),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Activation('softmax')
])

model.summary()

# 2.モデルのコンパイル
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])


#モデルを学習させる
model.fit(x_train, y_train, epochs=3, verbose=1,
          validation_data=(x_test, y_test))

#テスト関数


@tf.function
def test_step(x):
  model(x)


#test_stepのグラフ化により高速化
test_step_int = test_step.get_concrete_function(
    tf.TensorSpec(shape=(None, 784), dtype=tf.float64)
)

#print(test_step_int.function_def)

#モデルの精度
score = model.evaluate(x_test, y_test, verbose=0)
print('score:', score[0])
print('accuracy:', score[1])

start = time.perf_counter()
n_loop = 5
num = 0
for n in range(n_loop):
    for x in test_ds:
        test_step_int(x)
        num += 1
print('-' * 30)
print('elapsed time for {} prediction {} [msec]'.format(
    num / n_loop, (time.perf_counter() - start) * 1000 / n_loop))
