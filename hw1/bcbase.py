import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.client import device_lib
import os
from os import listdir
from os.path import isfile, join
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class BaseBehaviourCloningModel:

    def __init__(self, env_name):
        self.env_name = env_name
        self.observations = None
        self.actions = None
        self._model = None
        self._X_train = None
        self._X_test = None
        self._y_train = None
        self._y_test = None

    def load_from_file(self, filepath, limit_size=200000):
        with open(filepath, "rb") as f:
            expert_data = pickle.loads(f.read())

        if self.observations is None:
            self.observations = expert_data['observations']
            self.actions = expert_data['actions'][:, 0]
        else:
            new_observations = expert_data['observations']
            new_actions = expert_data['actions'][:, 0]
            self.observations = np.vstack((new_observations, self.observations))
            self.actions = np.vstack((new_actions, self.actions))

        assert self.observations.shape[0] == self.actions.shape[0], "Shapes do not match"

        self.observations = self.observations[:limit_size]
        self.actions = self.actions[:limit_size]

        print("observations size: ", self.observations.shape)

    def load_from_dir(self, dir_path):
        files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
        for file in files:
            self.load_from_file(os.path.join(dir_path + '/', file))

    def train_test_split(self, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(self.observations, self.actions, test_size=test_size)
        return X_train, X_test, y_train, y_test

    def get_current_model(self):
        return self._model

    # Build the model
    def baseline_model(self, X_train, y_train):
        model = keras.Sequential([
            layers.Dense(1024, activation=tf.nn.relu, input_shape=[X_train.shape[1]]),
            layers.Dense(1024, activation=tf.nn.relu),
            layers.Dense(512, activation=tf.nn.relu),
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(y_train.shape[1])
        ])

        model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                      loss='mse',  # mean squared error
                      metrics=['mean_absolute_error', 'mean_squared_error'])
        return model

    # Build the model
    def shallow_model(self, X_train, y_train):
        n_inputs = X_train.shape[1]
        n_outputs = y_train.shape[1]

        model = keras.Sequential([
            layers.Dense(128,
                         activation=tf.nn.relu,
                         kernel_initializer='glorot_normal',
                         kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                         input_shape=(n_inputs,)),
            layers.Dense(128,
                         activation=tf.nn.relu,
                         kernel_initializer='glorot_normal',
                         kernel_regularizer=tf.keras.regularizers.l2(l=0.01)),
            layers.Dense(n_outputs)
        ])

        model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                      loss='mse',  # mean squared error
                      metrics=['mean_absolute_error', 'mean_squared_error'])
        return model


    def train(self, X_train, X_test, y_train, y_test, model=None, model_fn=None, epochs_number=1000,
              batch_size=512):
        print(device_lib.list_local_devices())

        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test
        num_training_samples = self._X_train.shape[0]

        if model is not None:
            self._model = model
        elif model_fn == None:
            self._model = self.baseline_model(X_train, y_train)
        else:
            self._model = model_fn(X_train, y_train)

        # model = self.build_model(X_train, y_train)
        print(self._model.summary())

        # Use the Datasets API to scale to large datasets or multi-device training
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        dataset = dataset.batch(batches_size).repeat()

        val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        val_dataset = val_dataset.batch(batches_size).repeat()

        tbCallBack = keras.callbacks.TensorBoard(log_dir="./graphs", histogram_freq=0, write_graph=True,
                                                 write_images=True)

        history = self._model.fit(dataset, epochs=epochs_number, steps_per_epoch=(num_training_samples//batches_size),
                                  validation_data=val_dataset,
                                  validation_steps=3
                                  , callbacks=[tbCallBack])

        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        print(hist.tail())

        def plot_history(history):
            hist = pd.DataFrame(history.history)
            hist['epoch'] = history.epoch

            plt.figure()
            plt.xlabel('Epoch')
            plt.ylabel('Mean Abs Error [MPG]')
            plt.plot(hist['epoch'], hist['mean_absolute_error'],
                     label='Train Error')
            plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
                     label='Val Error')
            plt.ylim([0, 5])
            plt.legend()

            plt.figure()
            plt.xlabel('Epoch')
            plt.ylabel('Mean Square Error [$MPG^2$]')
            plt.plot(hist['epoch'], hist['mean_squared_error'],
                     label='Train Error')
            plt.plot(hist['epoch'], hist['val_mean_squared_error'],
                     label='Val Error')
            plt.ylim([0, 20])
            plt.legend()
            plt.show()

        plot_history(history)

    def save_model(self, filename):
        if os.path.exists(filename):
            os.remove(filename)
        self._model.save(filename)

    def load_model(self, filename):
        self._model = tf.keras.models.load_model(filename)
        self._model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                          loss='mse',
                          metrics=['mean_absolute_error', 'mean_squared_error'])
        print(self._model.summary())

    def predict(self, observation):
        action = self._model.predict(observation)
        return action
        # return action[:, None].T
