import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers
import numpy as np
import os
from maze_env import Maze
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# (xs, ys), (x_val, y_val) = datasets.mnist.load_data()


# xs = tf.convert_to_tensor(xs, dtype=tf.float32) / 255
# ys = tf.convert_to_tensor(ys, dtype=tf.int32)
# ys = tf.one_hot(ys, depth=10)

# tf.pad

# train_dataset = tf.data.Dataset.from_tensor_slices((xs, ys))
# train_dataset = train_dataset.batch(200)

# # db = tf.data.Dataset.from_tensor_slices((xs, ys)).batch(64)

# model = keras.Sequential([
#     layers.Dense(512, activation='relu'),
#     layers.Dense(256, activation='relu'),
#     layers.Dense(10)])
    

# optimizer = optimizers.SGD(learning_rate=0.001)

# def train_epoch(epoch):
#     for step, (x, y) in enumerate(train_dataset):
        
#         with tf.GradientTape() as tape:
#             x = tf.reshape(x, (-1, 28*28))
#             out = model(x)
#             loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]

#         grads = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(grads, model.trainable_variables))

#         if step % 100 == 0:
#             print(epoch, step, "loss:",  loss.numpy())              
        
    

# def train():
#     for epoch in range(30):
#         train_epoch(epoch)    

# if __name__ == '__main__':
#     x = tf.random.normal([2, 3])
#     model = keras.Sequential([keras.layers.Dense(2, activation='relu'),
#             keras.layers.Dense(2, activation='relu'),
#             keras.layers.Dense(2, activation='relu')])
    
#     model.build(input_shape=[None, 4])
#     model.summary()

#     for p in model.trainable_variables:
#         print(p.name, p.shape)

# a = tf.linspace(-10., 10., 18)
# with tf.GradientTape() as tape:
#     tape.watch(a)
#     y = tf.nn.relu(a)

# grads = tape.gradient(y, [a])

# print(grads)   


#x = tf.constant([[1, 2, 4], [3, 4, 4]])
#w = tf.constant([[1], [2]])

#print(x)
image = np.array([[
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 2, 1, 0],
    [0, 0, 2, 2, 0, 1, 0],
    [0, 1, 1, 0, 2, 1, 0],
    [0, 0, 2, 1, 1, 0, 0],
    [0, 2, 1, 1, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
]], dtype=np.float32)

#print(np.shape(image))
#image = np.expand_dims(image, axis=-1)
#print(image)

# a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
# for i in range(a.shape[1]):
#     min_ = np.min(a[:, i])
#     max_ = np.max(a[:, i])
#     print(min_, max_)
#     a[:, i] = (a[:, i] - min_) / (max_ - min_)

# print(a[:, 2])
# print(a)

# env = Maze()
# while True:
#     env.render()
# t = np.array([22, 33])
# print(t)

a = tf.random.normal([2, 3, 4])
print(a)