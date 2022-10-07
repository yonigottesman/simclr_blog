import tensorflow as tf
import tensorflow_datasets as tfds

from augmentation import preprocess_for_train


def stl10_ds(global_batch_size, jitter_strength=0.5, image_size=96):
    train_ds = tfds.load("stl10", split="unlabelled", as_supervised=True, shuffle_files=True)
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(10 * global_batch_size)
    train_ds = train_ds.map(
        lambda image, label: tf.image.convert_image_dtype(image, tf.float32),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    train_ds = train_ds.map(
        lambda image: (
            preprocess_for_train(image, image_size, image_size, jitter_strength=jitter_strength),
            preprocess_for_train(image, image_size, image_size, jitter_strength=jitter_strength),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    train_ds = train_ds.batch(global_batch_size, drop_remainder=True)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds
