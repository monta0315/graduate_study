import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np

def create_model():
    inputs = layers.Input((784,))
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(10, activation="softmax")(x)
    return tf.keras.models.Model(inputs, x)

def main():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype(np.float32).reshape(-1, 784) / 255.0
    X_test = X_test.astype(np.float32).reshape(-1, 784) / 255.0

    # tf.dataによるデータセットを作る（訓練データ）
    trainset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    trainset = trainset.shuffle(buffer_size=1024).batch(128)

    model = create_model()
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    acc = tf.keras.metrics.SparseCategoricalAccuracy()
    optim = tf.keras.optimizers.Adam()

    @tf.function # 高速化のためのデコレーター
    def train_on_batch(X, y):
        with tf.GradientTape() as tape:
            pred = model(X, training=True)
            loss_val = loss(y, pred)
        # backward
        graidents = tape.gradient(loss_val, model.trainable_weights)
        # step optimizer
        optim.apply_gradients(zip(graidents, model.trainable_weights))
        # update accuracy
        acc.update_state(y, pred)  # 評価関数に結果を足していく
        return loss_val

    # train
    for i in range(5):
        acc.reset_states()  # 評価関数の集積をリセット
        print("Epoch =", i)

        for step, (X, y) in enumerate(trainset):
            loss_val = train_on_batch(X, y)

            if step % 100 == 0:
                print(f"Step = {step}, Loss = {loss_val}, Total Accuracy = {acc.result()}")

    # テストデータ
    testset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    testset = testset.batch(128)

    acc.reset_states()
    for step, (X, y) in enumerate(testset):
        pred = model(X, training=False)
        acc.update_state(y, pred)
    print("Final test accuracy : ", acc.result().numpy()) # acc自体はSparseCategoricalAccuracyなのでresult()を呼び出す

if __name__ == "__main__":
    main()
