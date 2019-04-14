"""
Implementation of feed forward neural network in TensorFlow
for solving behaviour cloning problem of open ai gym environment.
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import time
import numpy as np


def create_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass


class TFModel:

    def __init__(self,
                 layers_dims=None,
                 batch_size=128,
                 learning_rate=0.001,
                 shuffle_train_data=True,
                 activation_function=tf.nn.relu):
        """ Initializes the model.
        :param train_data:
        :param test_data:
        :param shuffle_train_data:
        :param batch_size:
        :param learning_rate:
        :param layers_dims: structure of the neural network, numpy array. Ex.: [100, 200] represents neural net
                              with two hidden layers, which consist of 100 and 200 units respectively.
        """
        self.layers_dims = layers_dims
        self.shuffle_train_data = shuffle_train_data
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self._activation_function = activation_function
        self.global_step = tf.get_variable('global_step', initializer=tf.constant(0), trainable=False)

    def _import_data(self):
        """Creates datasets and iterators.
        """
        with tf.name_scope('data'):
            # create placeholders to feed NumPy arrays during the session run
            self.features_placeholder = tf.placeholder(self.test_data[0].dtype,
                                                       shape=[None, self.train_data[0].shape[1]])
            self.labels_placeholder = tf.placeholder(self.test_data[1].dtype, shape=[None, self.train_data[1].shape[1]])

            # create train dataset through placeholders
            self.train_dataset = tf.data.Dataset.from_tensor_slices(
                (self.features_placeholder, self.labels_placeholder))
            if self.shuffle_train_data:
                self.train_dataset.shuffle(buffer_size=10000)
            self.train_dataset = self.train_dataset.batch(self.batch_size)

            # create test dataset through placeholders
            self.test_dataset = tf.data.Dataset.from_tensor_slices((self.features_placeholder, self.labels_placeholder))
            self.test_dataset = self.test_dataset.batch(self.batch_size)

            # create feedable iterator
            self.iterator = tf.data.Iterator.from_structure(self.train_dataset.output_types,
                                                            self.train_dataset.output_shapes)
            (self.input, self.output) = self.iterator.get_next()

            self.train_init = self.iterator.make_initializer(self.train_dataset)  # initializer for train_data
            self.test_init = self.iterator.make_initializer(self.test_dataset)  # initializer for test_data

    def add_data(self, train_data, test_data=None):
        """Adds data to datasets.
            Args:
              train_data: features, labels, tuple
              test_data:  features, labels, tuple
        """
        self.train_data = (np.vstack((train_data[0], self.train_data[0])),
                           np.vstack((train_data[1], self.train_data[1])))

        if test_data is not None:
            self.test_data = (np.vstack((test_data[0], self.test_data[0])),
                              np.vstack((test_data[1], self.test_data[1])))

        print('Train data size after adding: {0}'.format(len(self.train_data[0])))

    def _initialize_layer(self, n_current_layer, n_previous_layer, scope):
        """Initializes weights and biases of the neural network"""
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
            w = tf.get_variable('weights', dtype=tf.float32, shape=(n_current_layer, n_previous_layer),
                                initializer=tf.glorot_uniform_initializer())
            b = tf.get_variable('biases', dtype=tf.float32, shape=(n_current_layer, 1),
                                initializer=tf.zeros_initializer())
            return w, b

    def _forward_propagation(self, input):
        """Defines inference model"""
        x = input
        n_l = len(self.layers_dims)
        for l in range(1, n_l):
            w, b = self._initialize_layer(self.layers_dims[l], self.layers_dims[l-1], 'layer{0}'.format(l))
            x = tf.matmul(w, x) + b
            if l is not n_l - 1:
                x = self._activation_function(x)
        return x

    def _create_loss(self):
        """Defines loss function"""
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.square(self.prediction - tf.transpose(self.output)), name='loss')

    def _create_optimizer(self):
        """Defines optimizer"""
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate) \
            .minimize(self.loss, global_step=self.global_step)

    def _create_summaries(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def build_graph(self, train_data, test_data):
        """Builds the graph for model."""
        self.train_data = train_data
        self.test_data = test_data
        self._import_data()
        self.prediction = self._forward_propagation(tf.transpose(self.input))
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()

    def _train_one_epoch(self, epoch, sess, init, writer, saver, step):
        # start_time = time.time()

        sess.run(init, feed_dict={self.features_placeholder: self.train_data[0],
                                  self.labels_placeholder: self.train_data[1]})
        total_loss = 0
        n_batches = 0
        try:
            while True:
                loss_batch, _, summary = sess.run([self.loss, self.optimizer, self.summary_op])
                writer.add_summary(summary, global_step=step)
                step += 1
                total_loss += loss_batch
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass

        saver.save(sess, 'checkpoints/tfmodel', step)
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss / n_batches))
        # print('Took {0} seconds'.format(time.time()-start_time))
        return step

    def _eval_once(self, epoch, sess, init, writer, step):
        # start_time = time.time()
        sess.run(init, feed_dict={self.features_placeholder: self.test_data[0],
                                  self.labels_placeholder: self.test_data[1]})

        total_loss = 0
        n_batches = 0
        try:
            while True:
                loss_batch, summary = sess.run([self.loss, self.summary_op])
                writer.add_summary(summary, global_step=step)
                total_loss += loss_batch
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass

        print('Validation loss at epoch {0}: {1}'.format(epoch, total_loss / n_batches))
        # print('Took: {0} seconds'.format(time.time() - start_time))

    def train(self, n_epochs):
        """
        The train function alternates between training one epoch and evaluating
        """
        create_dir('checkpoints')
        create_dir('checkpoints/tfmodel')

        writer = tf.summary.FileWriter('./graphs/{0}'.format(str(int(time.time()))), tf.get_default_graph())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver()
            # ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/tfmodel/checkpoint'))
            # if ckpt and ckpt.model_checkpoint_path:
            #    saver.restore(sess, ckpt.model_checkpoint_path)
            step = self.global_step.eval()

            for epoch in range(n_epochs):
                step = self._train_one_epoch(epoch, sess, self.train_init, writer, saver, step)
                self._eval_once(epoch, sess, self.test_init, writer, step)
                writer.flush()

            # saver.save(sess, 'models/tfmodel/model.ckpt')

        writer.close()

    def restore_model(self, sess):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('./checkpoints')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

    def predict(self, input, sess):
        return sess.run(self.prediction, feed_dict={self.input: input})
