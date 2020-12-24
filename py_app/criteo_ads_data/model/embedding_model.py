import logging
import math
import pickle
from typing import Union

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Flatten
from tensorflow.keras.models import Model

from layers.processing_layer import CustomNormalizationLayer, \
    CustomStringLookupLayer


class EmbeddingModel(Model):

    def __init__(
            self,
            lst_features: str,
            layer_dir: str,
            embedding_dim_base: int = None,
            default_embedding_dim: int = 3,
            **kwargs
    ) -> None:

        super().__init__(**kwargs)

        # feature sub-model
        self.dict_submodel = dict()
        for feature in lst_features:
            logging.info(f'prep {feature} sub-model')
            tmp_processing_layer = self._deserializing_layer(
                feature=feature,
                layer_dir=layer_dir
            )

            # apply embedding layer if categorical feature
            if feature != 'numeric':
                emb_input_dim = tmp_processing_layer.vocab_size()
                if embedding_dim_base is None or embedding_dim_base == 0:
                    emb_output_dim = default_embedding_dim
                else:
                    log10_emb_input_dim = math.log(
                        emb_input_dim,
                        embedding_dim_base,
                    )
                    emb_output_dim = int(log10_emb_input_dim) + 1
                embedding_layer = Embedding(
                    input_dim=emb_input_dim,
                    output_dim=emb_output_dim,
                    input_length=1,
                )
                tmp_cat_model = tf.keras.Sequential([
                    tmp_processing_layer,
                    embedding_layer,
                    Flatten()
                ], name=f'embedding_model_{feature}')
                self.dict_submodel.update({feature: tmp_cat_model})

            # apply normalization layer to numeric feature
            else:
                tmp_numeric_model = tf.keras.Sequential([
                    tmp_processing_layer,
                ], name=f'normalized_model_{feature}')
                self.dict_submodel.update({feature: tmp_numeric_model})

        # dense model
        self.dense_model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ], name='dense_model')

    def call(
            self,
            inputs: tf.data.Dataset,
            training=None,
            mask=None,
    ) -> tf.Tensor:

        ls_tensor = list()
        for feature, submodel in self.dict_submodel.items():
            tmp_input = inputs[feature]
            tmp_tensor = submodel(tmp_input)
            ls_tensor.append(tmp_tensor)
        concat_tensor = tf.concat(ls_tensor, axis=1)
        return self.dense_model(concat_tensor)

    @staticmethod
    def _deserializing_layer(
            feature: str,
            layer_dir: str,
    ) -> Union[CustomNormalizationLayer, CustomStringLookupLayer]:

        if feature == 'numeric':
            custom_objects = {
                'CustomNormalizationLayer': CustomNormalizationLayer
            }
        else:
            custom_objects = {
                'CustomStringLookupLayer': CustomStringLookupLayer
            }

        # deserializing layer
        logging.info(f'deserializing layer - {feature}')
        pkl_output_path = f'{layer_dir}/processinglayer_{feature}.pkl'
        with open(pkl_output_path, 'rb') as f:
            serialized_layer = pickle.load(f)
        layer = tf.keras.layers.deserialize(
            serialized_layer,
            custom_objects=custom_objects
        )
        return layer
