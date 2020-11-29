import os
from typing import Tuple

import tensorflow as tf

from dataprep.tfrecord import tfrecord_decoder


def _is_validate(idx: int, _) -> bool:
    return idx % 5 == 0


def _is_train(idx: int, data: tf.data.Dataset) -> bool:
    return not _is_validate(idx, data)


def _recover(_, data: tf.data.Dataset) -> tf.data.Dataset:
    return data


def dataprep(
        data_path: str,
        data_filename: str,
        batch_size: int,
        shuffle_buffer: int = 10000,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:

    tf_autotune = tf.data.experimental.AUTOTUNE

    dataset = tf.data.TFRecordDataset(
        filenames=os.path.join(data_path, data_filename),
        num_parallel_reads=2,
    ) \
        .map(tfrecord_decoder, num_parallel_calls=tf_autotune) \
        .cache() \
        .batch(batch_size=batch_size) \
        .shuffle(shuffle_buffer, reshuffle_each_iteration=False) \
        .prefetch(buffer_size=tf_autotune)

    validate_dataset = dataset.enumerate() \
        .filter(_is_validate) \
        .map(_recover, num_parallel_calls=tf_autotune)

    train_dataset = dataset.enumerate() \
        .filter(_is_train) \
        .map(_recover, num_parallel_calls=tf_autotune)

    return train_dataset, validate_dataset
