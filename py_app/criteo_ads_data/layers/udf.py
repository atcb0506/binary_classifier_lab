import logging
import pickle

import tensorflow as tf

from layers.processing_layer import \
    CustomNormalizationLayer, CustomStringLookupLayer


def adapting_preprocssing_layer(
        feature: str,
        data: tf.data.Dataset,
        output_path: str,
) -> None:

    if feature == 'numeric':
        layer_obj = CustomNormalizationLayer(
            feature_key=feature
        )
    else:
        layer_obj = CustomStringLookupLayer(
            feature_key=feature
        )

    # adapting layer
    logging.info(f'adapting layer - {feature}')
    layer_obj.adapt(data)

    # serializing layer
    logging.info(f'serializing layer - {feature}')
    pkl_output_path = f'{output_path}/processinglayer_{feature}.pkl'
    serialized_layer = tf.keras.layers.serialize(layer_obj)
    with open(pkl_output_path, 'wb') as f:
        pickle.dump(serialized_layer, f)
