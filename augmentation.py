from functools import partial

import tensorflow as tf


# Copied from https://github.com/google-research/simclr/blob/2fc637bdd6a723130db91b377ac15151e01e4fc2/data_util.py#L252
def distorted_bounding_box_crop(
    image,
    bbox,
    min_object_covered=0.1,
    aspect_ratio_range=(0.75, 1.33),
    area_range=(0.05, 1.0),
    max_attempts=100,
):

    shape = tf.shape(image)
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        shape,
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True,
    )
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    image = tf.image.crop_to_bounding_box(image, offset_y, offset_x, target_height, target_width)

    return image


# Copied from https://github.com/google-research/simclr/blob/2fc637bdd6a723130db91b377ac15151e01e4fc2/data_util.py#L304
def crop_and_resize(image, height, width):

    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    aspect_ratio = width / height
    image = distorted_bounding_box_crop(
        image,
        bbox,
        min_object_covered=0.1,
        aspect_ratio_range=(3.0 / 4 * aspect_ratio, 4.0 / 3.0 * aspect_ratio),
        area_range=(0.08, 1.0),
        max_attempts=100,
    )
    return tf.image.resize([image], [height, width], tf.image.ResizeMethod.BICUBIC)[0]


def color_jitter(image, strength=1):
    brightness = 0.8 * strength
    contrast = 0.8 * strength
    saturation = 0.8 * strength
    hue = 0.2 * strength

    # apply 4 functions in random order
    perm = tf.random.shuffle(tf.range(4))
    for i in range(4):
        image = tf.switch_case(
            perm[i],
            branch_fns={
                0: lambda: tf.random.uniform([], tf.maximum(1.0 - brightness, 0), 1.0 + brightness) * image,
                1: lambda: tf.image.random_contrast(image, lower=1 - contrast, upper=1 + contrast),
                2: lambda: tf.image.random_saturation(image, lower=1 - saturation, upper=1 + saturation),
                3: lambda: tf.image.random_hue(image, max_delta=hue),
            },
        )
        image = tf.clip_by_value(image, 0, 1)

    return image


def random_apply(func, p, x):
    """Randomly apply function func to x with probability p."""
    return tf.cond(
        tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32), tf.cast(p, tf.float32)),
        lambda: func(x),
        lambda: x,
    )


# https://github.com/google-research/simclr/blob/2fc637bdd6a723130db91b377ac15151e01e4fc2/data_util.py#L328
def gaussian_blur(image, kernel_size, sigma, padding="SAME"):
    radius = tf.cast(kernel_size / 2, tf.int32)
    kernel_size = radius * 2 + 1
    x = tf.cast(tf.range(-radius, radius + 1), tf.float32)
    blur_filter = tf.exp(-tf.pow(x, 2.0) / (2.0 * tf.pow(tf.cast(sigma, tf.float32), 2.0)))
    blur_filter /= tf.reduce_sum(blur_filter)
    # One vertical and one horizontal filter.
    blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
    blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
    num_channels = tf.shape(image)[-1]
    blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
    blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
    expand_batch_dim = image.shape.ndims == 3
    if expand_batch_dim:
        # Tensorflow requires batched input to convolutions, which we can fake with
        # an extra dimension.
        image = tf.expand_dims(image, axis=0)
    blurred = tf.nn.depthwise_conv2d(image, blur_h, strides=[1, 1, 1, 1], padding=padding)
    blurred = tf.nn.depthwise_conv2d(blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)
    if expand_batch_dim:
        blurred = tf.squeeze(blurred, axis=0)
    return blurred


def preprocess_for_train(image, height, width, color_distort=True, crop=True, flip=True, blur=True, jitter_strength=1):
    if crop:
        image = random_apply(lambda image: crop_and_resize(image, height, width), p=1, x=image)
    if flip:
        image = tf.image.random_flip_left_right(image)
    if color_distort:
        image = random_apply(lambda image: partial(color_jitter, strength=jitter_strength)(image), p=0.8, x=image)
        image = random_apply(lambda image: tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 3]), p=0.2, x=image)
    if blur:
        sigma = tf.random.uniform([], 0.1, 2.0, dtype=tf.float32)
        image = random_apply(
            lambda image: gaussian_blur(image, kernel_size=height // 10, sigma=sigma, padding="SAME"), p=0.5, x=image
        )

    image = tf.reshape(image, [height, width, 3])
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image
