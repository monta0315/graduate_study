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
x_train = x_train.astype(np.float32).reshape(-1, 784) / 255.0
x_test = x_test.astype(np.float32).reshape(-1, 784) / 255.0

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

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=128)

@tf.function
def train_step(x, y):
  with tf.GradientTape() as tape:
    pred = model(x, training=True)
    loss_val = loss(y, pred)

  # backward
    graidents = tape.gradient(loss_val, model.trainable_weights)
    # step optimizer
    optim.apply_gradients(zip(graidents, model.trainable_weights))
    # update accuracy
    acc.update_state(y, pred)  # 評価関数に結果を足していく
    return loss_val


#グラフ出力用の変数を定数に置き換える処理をしているつもり
train_step_int = train_step.get_concrete_function(
    tf.TensorSpec(shape=((None),(None)),dtype=((tf.float32),(tf.uint8)))
)


#訓練データ作成
trainset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
tf.print(trainset)
trainset = trainset.shuffle(buffer_size=1024).batch(128)
tf.print(trainset)

#テストデータ作成
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = test_ds.batch(128)

""" @tf.function
def test_step(x):
  model(np.array([x]))
 """
n_loop = 1
start=time.perf_counter()
for n in range(n_loop):
  acc.reset_states()  # 評価関数の集積をリセット
  print("Epoch =", n)

  for step, (X, y) in enumerate(trainset):
      loss_val = train_step(X, y)
      if step % 100 == 0:
        print(
            f"Step = {step}, Loss = {loss_val}, Total Accuracy = {acc.result()}")

#学習後のモデルをgraph化してエクスポートしたい
train_step_int(trainset)
train_step.function_def


#trainsetのデータ型確認
#tf.print(trainset)

""" for x in x_test:
  model.predict(np.array([x]),batch_size=128,verbose=1) """

print('elapsed time for {} prediction {} [msec]'.format(
    len(x_test), (time.perf_counter()-start) * 1000 / n_loop))

print()
