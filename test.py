import tensorflow as tf

# MNISTデータセットを使用
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#よくわからんけど予測関数を直接実装する際に必要っぽい？
test_ds = tf.data.Dataset.from_tensor_slices((x_test)).batch(1)
print(len(test_ds))
