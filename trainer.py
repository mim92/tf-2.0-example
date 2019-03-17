import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

import time
from tqdm import tqdm

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


class EagerTrainer(object):
    def __init__(self):
        self.model = MyModel()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(self, image, label):
        with tf.GradientTape() as tape:
            predictions = self.model(image)
            loss = self.loss_object(label, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(label, predictions)

    @tf.function
    def test_step(self, image, label):
        predictions = self.model(image)
        t_loss = self.loss_object(label, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(label, predictions)

    def train(self, epochs, training_data, test_data):
        template = 'Epoch {}, Loss: {:.5f}, Accuracy: {:.5f}, Test Loss: {:.5f}, Test Accuracy: {:.5f}, elapsed_time {:.5f}'

        for epoch in range(epochs):
            start = time.time()
            for image, label in tqdm(training_data):
                self.train_step(image, label)
            elapsed_time = time.time() - start

            for test_image, test_label in test_data:
                self.test_step(test_image, test_label)

            print(template.format(epoch + 1,
                                  self.train_loss.result(),
                                  self.train_accuracy.result() * 100,
                                  self.test_loss.result(),
                                  self.test_accuracy.result() * 100,
                                  elapsed_time))


class KerasTrainer(object):
    def __init__(self):
        self.model = MyModel()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])

    def train(self, epochs, training_data, test_data):
        self.model.fit(training_data, epochs=epochs, validation_data=test_data)
