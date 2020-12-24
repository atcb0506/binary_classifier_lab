import glob
import logging

from typing import Tuple, Union

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
        batch_size: int,
        test_split: bool = True,
) -> Union[Tuple[tf.data.Dataset, tf.data.Dataset], tf.data.Dataset]:

    # get the list of input files
    input_files = glob.glob(f'{data_path}/*.tfrecord.gz')
    logging.info(f'Loaded: {input_files}')

    tf_autotune = tf.data.experimental.AUTOTUNE
    dataset = tf.data.TFRecordDataset(
        filenames=input_files,
        compression_type='GZIP',
        num_parallel_reads=2,
    ) \
        .map(tfrecord_decoder, num_parallel_calls=tf_autotune) \
        .cache() \
        .batch(batch_size=batch_size) \
        .prefetch(buffer_size=tf_autotune)

    if not test_split:
        return dataset

    validate_dataset = dataset.enumerate() \
        .filter(_is_validate) \
        .map(_recover, num_parallel_calls=tf_autotune)

    train_dataset = dataset.enumerate() \
        .filter(_is_train) \
        .map(_recover, num_parallel_calls=tf_autotune)

    return train_dataset, validate_dataset
