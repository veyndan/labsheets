import collections
import tensorflow as tf
import os
import os.path
import numpy as np
import pandas as pd

session = tf.Session()

data = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        sep=",",
        names=["sepal_length", "sepal_width", "petal_length", "petal_width", "iris_class"]
)

np.random.seed(0)

data = data.sample(frac=1).reset_index(drop=True)

all_x = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
all_y = pd.get_dummies(data.iris_class)

n_x = len(all_x.columns)
n_y = len(all_y.columns)

train_x, train_y = all_x[:100], all_y[:100]
test_x, test_y = all_x[100:], all_y[100:]

x = tf.placeholder(tf.float32, shape=[None, n_x])
y = tf.placeholder(tf.float32, shape=[None, n_y])

W = tf.get_variable("W", shape=[n_x, n_y], initializer=tf.zeros_initializer())
b = tf.get_variable("b", shape=[n_y], initializer=tf.zeros_initializer())

prediction = tf.nn.softmax(tf.matmul(x, W) - b)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), axis=1))

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, axis=1), tf.argmax(y, axis=1)), tf.float32))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

session.run(tf.global_variables_initializer())

for epoch in range(1000):
    session.run(optimizer, feed_dict={x: train_x, y: train_y})

    if epoch % 100 == 0:
        acc = session.run(accuracy, feed_dict={x: test_x, y: test_y})
        print("Accuracy of Perceptron at epoch {} is {}".format(epoch, accuracy))
