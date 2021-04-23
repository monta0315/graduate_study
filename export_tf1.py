import tensorflow as tf
import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data
import os

f_model="./models_tf"

#ディレクトリを作成
if not os.path.isdir(f_model):
  os.makedirs(f_model)

# config setting
imageDim = 784
outputDim = 10

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# None means that a dimension can be any of any length
x = tf.compat.v1.placeholder(tf.float32, [None, imageDim], name="input")
# 784-dimensional image vectors by it to produce 10-dimensional vectors
W = tf.Variable(tf.zeros([imageDim, outputDim]),
                dtype=tf.float32, name="Weight")
# a shape of [10]
b = tf.Variable(tf.zeros([outputDim]), dtype=tf.float32, name="bias")
# softmax
y = tf.nn.softmax(tf.matmul(x, W)+b, name="softmax")
# print(x, W, b, y)

# input correct answers
y_ = tf.compat.v1.placeholder(tf.float32, [None, outputDim])

# cross-entropy
cross_entropy = tf.reduce_mean(
    -tf.reduce_sum(y_ * tf.math.log(y), reduction_indices=[1])
)

train_step = tf.compat.v1.train.GradientDescentOptimizer(
    0.5).minimize(cross_entropy)

#変数を保存、復元する目的で
saver = tf.train.Saver()


print("tensorflow network already prepare done...")

with tf.compat.v1.Session() as sess:
    print('Training new network...')
    sess.run(tf.compat.v1.global_variables_initializer())

    #training
    for i in range(1000):
        if i % 100 == 0:
            print("iteration num :", i)
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    #チェックポイントとグラフをpbファイルとして出力する
    saver.save(sess, f_model + "/model.ckpt")
    tf.train.write_graph(sess.graph.as_graph_def(),
    f_model, "graph.pb")


    # Testing
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    start = time.perf_counter()
    print(
        "accuracy : ",
        sess.run(accuracy,feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    print('elapsed time {} [msec]'.format(
         (time.perf_counter()-start) * 1000))

    start = time.perf_counter()
    n_loop = 5
    for n in range(n_loop):
            [sess.run(y, feed_dict={x: np.array([test_x])})
             for test_x in mnist.test.images]
    print('elapsed time for {} prediction {} [msec]'.format(
            len(mnist.test.images), (time.perf_counter()-start) * 1000 / n_loop))
