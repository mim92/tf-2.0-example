import argparse

import tensorflow as tf
import tensorflow_datasets as tfds

from trainer import EagerTrainer, KerasTrainer


def convert_types(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label


def main():
    parser = argparse.ArgumentParser(description='Train Example')
    parser.add_argument('--trainer', type=str, default='eager')

    args = parser.parse_args()

    dataset, info = tfds.load('mnist', with_info=True, as_supervised=True)
    mnist_train, mnist_test = dataset['train'], dataset['test']

    mnist_train = mnist_train.map(convert_types).shuffle(10000).batch(32)
    mnist_test = mnist_test.map(convert_types).batch(32)

    if args.trainer.lower() == 'eager':
        trainer = EagerTrainer()
    else:
        trainer = KerasTrainer()
    trainer.train(epochs=5, training_data=mnist_train, test_data=mnist_test)


if __name__ == '__main__':
    main()
