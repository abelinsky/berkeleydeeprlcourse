"""
Implementation of feed forward neural network with Keras.
for solving behaviour cloning problem of open ai gym environment.
"""
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import TFModel as tfmodel
from TFModel import TFModel
import pandas as pd
import time
import numpy as np


class KerasModel(TFModel):

    def build_graph(self, train_data, test_data):
        """Builds the graph for model."""
        self.train_data = train_data
        self.test_data = test_data
        super(KerasModel, self)._import_data()

        # inputs = keras.Input(shape=(self.train_data[0].shape[1], 1))

        inputs = keras.Input(shape=(self.layers_dims[0],))

        prediction = inputs
        num_layers = len(self.layers_dims)
        for l in range(1, num_layers):
            activation = None if l is num_layers-1 else self._activation_function
            prediction = layers.Dense(self.layers_dims[l], activation=activation)(prediction)

        self._model = tf.keras.Model(inputs=inputs, outputs=prediction)
        self._model.compile(optimizer=tf.train.AdamOptimizer(self.learning_rate),
                            loss='mse',
                            metrics=['mean_squared_error'])

    def train(self, n_epochs):
        log_dir = './graphs/{0}'.format(str(int(time.time())))
        tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True,
                                                 write_images=False, batch_size=self.batch_size)

        num_training_samples = len(self.train_data[0])
        history = self._model.fit(x=self.train_data[0], y=self.train_data[1], epochs=n_epochs,
                                  steps_per_epoch=(num_training_samples // self.batch_size),
                                  # batch_size=self.batch_size,
                                  verbose=True,
                                  validation_data=self.test_data, validation_steps=1, callbacks=[tbCallBack])
        self._save_model()

        # hist = pd.DataFrame(history.history)
        # hist['epoch'] = history.epoch
        # print(hist.tail())

    def _save_model(self):
        tfmodel.create_dir('models')
        tfmodel.create_dir('./models/keras')
        filepath = './models/keras/kerasmodel.h5'
        # if os.path.exists(filepath):
        #    os.remove(filepath)
        self._model.save(filepath, overwrite=True)

    def restore_model(self, sess):
        filepath = './models/keras/kerasmodel.h5'
        self._model = tf.keras.models.load_model(filepath, compile=True)
        print(self._model.summary())

    def predict(self, input, sess):
        res = self._model.predict(input)
        res = np.array(res).squeeze()
        return res.reshape((len(res), 1))
