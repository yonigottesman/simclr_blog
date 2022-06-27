import keras_cv
import tensorflow as tf
import tensorflow_datasets as tfds

from augmentation import preprocess_for_train


def imagenette_tuples(image_size, batch_size):
    ds = tfds.load("imagenette/160px-v2")
    train_ds = ds["train"]
    train_ds = train_ds.shuffle(batch_size * 10)
    train_ds = train_ds.repeat()
    train_ds = train_ds.map(
        lambda sample: tf.image.convert_image_dtype(sample["image"], tf.float32),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    train_ds = train_ds.map(
        lambda image: (
            preprocess_for_train(image, image_size, image_size),
            preprocess_for_train(image, image_size, image_size),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds


def stl10_unsupervized(batch_size, jitter_strength=0.5, image_size=96):
    train_ds = tfds.load("stl10", split="unlabelled", as_supervised=True, shuffle_files=True)
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(10 * batch_size)
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
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds


def stl10_supervised(batch_size):
    train_ds = tfds.load("stl10", split="train", as_supervised=True, shuffle_files=True)
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(10 * batch_size)
    train_ds = train_ds.map(
        lambda image, label: (tf.image.convert_image_dtype(image, tf.float32), label),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    rand_augment = keras_cv.layers.RandAugment(value_range=(0, 1), augmentations_per_image=3, magnitude=0.3)
    train_ds = train_ds.map(
        lambda image, label: (rand_augment(image), label),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    test_ds = tfds.load("stl10", split="test", as_supervised=True, shuffle_files=True)
    test_ds = test_ds.cache()
    test_ds = test_ds.map(
        lambda image, label: (tf.image.convert_image_dtype(image, tf.float32), label),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )

    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, test_ds
