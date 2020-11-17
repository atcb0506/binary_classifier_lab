from typing import Tuple

import tensorflow as tf

from config import COLUMN_DEFAULTS, COLUMNS, LBL_COLUMN, NUM_COLUMNS
from loader.faeture_extraction import FeaturesExtraction


def _is_validate(
        idx: int,
        _
) -> bool:

    return idx % 5 == 0


def _is_train(
        idx: int,
        data: tf.data.Dataset
) -> bool:

    return not _is_validate(idx, data)


def _recover(
        _,
        data: tf.data.Dataset
) -> tf.data.Dataset:
    return data


def dataprep(
        data_path: str,
        batch_size: int
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:

    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=data_path,
        batch_size=batch_size,
        num_epochs=1,
        column_defaults=COLUMN_DEFAULTS,
        column_names=COLUMNS,
        label_name=LBL_COLUMN[0],
        field_delim='\t',
        shuffle=True,
        header=False,
    )\
        .shuffle(10, reshuffle_each_iteration=False)

    packed_dataset = dataset.map(FeaturesExtraction(num_col=NUM_COLUMNS))

    validate_dataset = packed_dataset.enumerate() \
        .filter(_is_validate) \
        .map(_recover)

    train_dataset = packed_dataset.enumerate() \
        .filter(_is_train) \
        .map(_recover)

    return train_dataset, validate_dataset
