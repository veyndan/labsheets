import collections
import tensorflow as tf
import os
import os.path
import numpy as np
import pandas as pd

g = tf.get_default_graph()
with g.as_default():
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

    # Number of nodes per layer
    ns = [n_x, 10, 20, 10, n_y]

    x = tf.placeholder(tf.float32, shape=[None, ns[0]])
    y = tf.placeholder(tf.float32, shape=[None, ns[-1]])

    W1 = tf.get_variable("W1", shape=[ns[0], ns[1]], initializer=tf.truncated_normal_initializer(stddev=0.1))
    b1 = tf.get_variable("b1", shape=[ns[1]], initializer=tf.constant_initializer(0.1))
    h1 = tf.nn.relu(tf.matmul(x, W1) + b1)

    W2 = tf.get_variable("W2", shape=[ns[1], ns[2]], initializer=tf.truncated_normal_initializer(stddev=0.1))
    b2 = tf.get_variable("b2", shape=[ns[2]], initializer=tf.constant_initializer(0.1))
    h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

    W3 = tf.get_variable("W3", shape=[ns[2], ns[3]], initializer=tf.truncated_normal_initializer(stddev=0.1))
    b3 = tf.get_variable("b3", shape=[ns[3]], initializer=tf.constant_initializer(0.1))
    h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)

    W4 = tf.get_variable("W4", shape=[ns[3], ns[4]], initializer=tf.truncated_normal_initializer(stddev=0.1))
    b4 = tf.get_variable("b4", shape=[ns[4]], initializer=tf.constant_initializer(0.1))
    h4 = tf.nn.relu(tf.matmul(h3, W4) + b4)

    prediction = h4

    with tf.name_scope('loss'):
        cost = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=prediction, scope="Cost_Function")
        tf.summary.scalar('loss', cost)

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, axis=1), tf.argmax(y, axis=1)), tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(cost)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./logs/train')
    test_writer = tf.summary.FileWriter('./logs/test')

    session.run(tf.global_variables_initializer())

    for epoch in range(3000):
        train_summary, _ = session.run([merged, optimizer], feed_dict={x: train_x, y: train_y})

        if epoch % 100 == 0:
            test_summary, acc = session.run([merged, accuracy], feed_dict={x: test_x, y: test_y})
            print("Accuracy of my first dnn at epoch {} is {}".format(epoch, acc))

        train_writer.add_summary(train_summary, epoch)
        test_writer.add_summary(test_summary, epoch)

