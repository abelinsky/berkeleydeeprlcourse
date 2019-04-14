"""
Implements behaviour cloning for OpenAI  Gym's environments.
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import math


class BehaviourCloningModel:

    def __init__(self, observations, actions):
        self._observations = observations
        self._actions = actions

    def random_mini_batches(self, X, Y, mini_batch_size=64, seed=0):
        """
        Creates a list of random minibatches from (X, Y)

        Arguments:
        X -- input data, of shape (input size, number of examples)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        mini_batch_size - size of the mini-batches, integer
        seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

        Returns:
        mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
        """

        m = X.shape[1]  # number of training examples
        mini_batches = []
        np.random.seed(seed)

        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(
            m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches

    def create_placeholders(self, n_x, n_y):
        """
        Create placeholders for TensorFlow session.

        :param n_x: scalar, dimensionality of observations states
        :param n_y: scalar, dimensionality of actions states

        :return:
        X -- placeholder for the data input (OBSERVATIONS), of shape [n_x, None] and dtype "float"
        Y -- placeholder for the input labels (ACTIONS), of shape [n_y, None] and dtype "float"
        """
        X = tf.placeholder(tf.float32, shape=[n_x, None], name="Observations")
        Y = tf.placeholder(tf.float32, shape=[n_y, None], name="Actions")

        return X, Y

    def initialize_params(self, n_x, n_y):
        """
        Initializes parameters to build neural network.

        :param n_x: scalar, dimensionality of observations states
        :param n_y: scalar, dimensionality of actions states

        :return: parameters, dictionary
        """

        W1 = tf.get_variable('W1', [128, n_x], initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable('b1', [128, 1], initializer=tf.zeros_initializer())
        W2 = tf.get_variable('W2', [n_y, 128], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable('b2', [n_y, 1], initializer=tf.zeros_initializer())

        parameters = {'W1': W1,
                      'b1': b1,
                      'W2': W2,
                      'b2': b2}

        return parameters

    def forward_propagation(self, X, parameters):
        # Retrieve the parameters
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)

        return Z2

    def compute_cost(self, Y_pred, Y):
        mse = tf.reduce_mean(0.5 * tf.square(Y_pred - Y))
        return mse

    def train(self, X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
              num_epochs=1000, minibatch_size=32):

        ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
        (n_x, m) = X_train.shape  # (n_x: input size, m : number of examples in the train set)
        n_y = Y_train.shape[0]
        costs = []

        X, Y = self.create_placeholders(n_x, n_y)
        parameters = self.initialize_params(n_x, n_y)

        Z = self.forward_propagation(X, parameters)
        cost = self.compute_cost(Z, Y)
        tf.summary.scalar("cost", cost)

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        # Merge all the summaries and write them out to log
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./graphs", tf.get_default_graph())  # before running session

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(num_epochs):
                epoch_cost = 0
                num_minibatches = int(m / minibatch_size)
                minibatches = self.random_mini_batches(X_train, Y_train, minibatch_size)

                for i, minibatch in enumerate(minibatches):
                    (minibatch_X, minibatch_Y) = minibatch
                    summary, _, minibatch_cost = sess.run([merged, optimizer, cost],
                                                 feed_dict={X: minibatch_X, Y: minibatch_Y})
                    if epoch % 5 == 0:
                        writer.add_summary(summary)

                    epoch_cost += minibatch_cost / num_minibatches

                if epoch % 100 == 0:
                    print("Cost after epoch {epoch} is {epoch_cost}".format(epoch=epoch, epoch_cost=epoch_cost))
                if (epoch % 5 == 0):
                    costs.append(epoch_cost)

            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

            # lets save the parameters in a variable
            parameters = sess.run(parameters)
            print("Parameters have been trained!")

        writer.close()

        return parameters
