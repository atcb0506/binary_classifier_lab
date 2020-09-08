import tensorflow as tf

from config import CAT_COLUMNS
from layers.processing_layer import DataProcessingLayer


def build_data_processing_layer(
        dataset: tf.data.Dataset
) -> DataProcessingLayer:

    layer = DataProcessingLayer(
        ls_cat_col=CAT_COLUMNS,
        num_col='numeric',
        name='data_processing_layer'
    )
    layer.adapt(dataset)

    return layer
