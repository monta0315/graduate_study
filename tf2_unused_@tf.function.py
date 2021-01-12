import tensorflow as tf
import os
import time
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import tensorflow.keras.layers as layers
import numpy as np
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

#tf2.0なのに@tf.functionを使用していないので死ぬほど時間かかります

#params
imageDim = 784
outputDim = 10
f_model="tf2_unused_@tf.function"

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

full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
    x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
)

# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="./frozen_models",
                  name="frozen_graph_tf2.pb",
                  as_text=True)


if not os.path.isdir(f_model):
  os.makedirs(f_model)
  model.save(f_model+'/my_model')


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
  for x in x_test:
    model.predict(np.array([x]),batch_size=128,verbose=1)

print('elapsed time for {} prediction {} [msec]'.format(
    len(x_test), (time.perf_counter()-start) * 1000 / n_loop))
