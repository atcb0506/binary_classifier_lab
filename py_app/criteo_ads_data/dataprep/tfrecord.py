import csv
from typing import Dict, Tuple, List, Generator

import tensorflow as tf

from config import COLUMNS, LBL_COLUMN, NUM_COLUMNS


def _bytes_feature(value: str) -> tf.train.Feature:
    if value == '':
        bytelist = tf.train.BytesList(value=['thisisdefault'.encode('utf-8')])
        return tf.train.Feature(bytes_list=bytelist)
    bytelist = tf.train.BytesList(value=[value.encode('utf-8')])
    return tf.train.Feature(bytes_list=bytelist)


def _int64_feature(value: str) -> tf.train.Feature:
    if value == '':
        intlist = tf.train.Int64List(value=[0])
        return tf.train.Feature(int64_list=intlist)
    intlist = tf.train.Int64List(value=[int(value)])
    return tf.train.Feature(int64_list=intlist)


def _combine_func(idx, x) -> tf.train.Feature:
    if idx < 14:
        return _int64_feature(x)
    return _bytes_feature(x)


def _encoder(row_data: List[str]) -> str:
    lst_value = [_combine_func(idx, x) for idx, x in enumerate(row_data)]
    tf_feature = tf.train.Features(feature=dict(zip(COLUMNS, lst_value)))
    tf_example = tf.train.Example(features=tf_feature) \
        .SerializeToString()
    return tf_example


def _read_csv(
        data_path: str,
        delimiter: str,
        nrow: int,
) -> Generator[str, None, None]:
    with open(data_path, 'r') as f:
        data = csv.reader(f, delimiter=delimiter)
        for idx, row in enumerate(data):
            if nrow == -1:
                yield _encoder(row)
            elif idx >= nrow:
                break
            yield _encoder(row)


def tfrecord_writer(
        input_path: str,
        tfrecord_path: str,
        delimiter: str,
        nrow: int,
) -> None:
    with tf.io.TFRecordWriter(tfrecord_path) as wf:
        for record in _read_csv(input_path, delimiter, nrow):
            wf.write(record)


def tfrecord_decoder(record) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    lst_type = [tf.io.FixedLenFeature([], dtype=tf.int64)] * 14 \
        + [tf.io.FixedLenFeature([], dtype=tf.string)] * 26
    dict_tf = tf.io.parse_single_example(
        record,
        dict(zip(COLUMNS, lst_type))
    )
    y = dict_tf.pop(LBL_COLUMN[0])
    numeric = [dict_tf.pop(col) for col in NUM_COLUMNS]
    numeric = tf.stack(numeric, axis=-1)
    dict_tf['numeric'] = numeric

    return dict_tf, y
