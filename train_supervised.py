import argparse

import tensorflow as tf

from data import stl10_supervised
from model import supervised_model


# https://github.com/matanby/keras-examples/blob/master/stl10.ipynb
def train():
    train_ds, test_ds = stl10_supervised(batch_size=16)

    model = supervised_model(96)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    model.compile(optimizer=optimizer, loss=loss, run_eagerly=True, metrics=metrics)

    model.fit(
        train_ds,
        epochs=100,
        validation_data=test_ds,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to config.yaml")
    train()
