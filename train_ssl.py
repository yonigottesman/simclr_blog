import argparse

import tensorflow as tf

from data import stl10_unsupervized
from loss import SimCLRLoss
from model import contrastive_model


def train():
    train_ds = stl10_unsupervized(batch_size=64)

    model = contrastive_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    loss = SimCLRLoss()

    model.compile(optimizer=optimizer, loss=loss, run_eagerly=False)

    model.fit(
        train_ds,
        epochs=1000,
        callbacks=[tf.keras.callbacks.ModelCheckpoint("checkpoint", monitor="loss", save_best_only=True)],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to config.yaml")
    train()
