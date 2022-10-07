import tensorflow as tf

from data import stl10_ds
from loss import SimCLRLoss
from model import contrastive_model


def train():
    ds = stl10_ds(global_batch_size=512)

    with tf.distribute.MirroredStrategy().scope():
        model = contrastive_model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        loss = SimCLRLoss()
        model.compile(optimizer=optimizer, loss=loss, run_eagerly=False)

    model.fit(
        ds,
        epochs=1000,
        callbacks=[tf.keras.callbacks.ModelCheckpoint("checkpoint", monitor="loss", save_best_only=True)],
    )


if __name__ == "__main__":
    train()
