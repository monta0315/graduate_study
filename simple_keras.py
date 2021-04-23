import tensorflow as tf
import time
import numpy as np
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.models import Sequential, model_from_yaml
from keras.utils import np_utils
from keras.layers.core import Dense, Flatten, Dropout, Activation
from keras.layers import InputLayer, MaxPooling2D, Convolution2D
import keras.backend.tensorflow_backend as KTF
from keras import backend as K
import argparse
import os
from keras.utils import plot_model

#config setting
batch_size = 128
nb_classes = 10


f_model = './models_keras'
model_filename = 'model.yaml'
weights_filename = 'model_weights.hdf5'


imageDim = 784
# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], imageDim)
x_test = x_test.reshape(x_test.shape[0], imageDim)
x_train, x_test = x_train / 255.0, x_test / 255.0

input_shape=(imageDim,)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#モデル定義
model = Sequential()
model.add(InputLayer(input_shape=(784,)))
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()

plot_model(model, to_file='model.png')

#モデルのコンパイル
model.compile(optimizer=SGD(lr=0.5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


#モデルの訓練
model.fit(x_train, Y_train,  epochs=5, verbose=1)

#モデルの精度
score = model.evaluate(x_test, Y_test, verbose=0)
print('score:', score[0])
print('accuracy:', score[1])

#推論の時間測定
start = time.perf_counter()
n_loop = 5
for n in range(n_loop):
        for x in x_test:
                model.predict(np.array([x]))

print('elapsed time for {} prediction {} [msec]'.format(
        len(x_test), (time.perf_counter()-start) * 1000 / n_loop))
