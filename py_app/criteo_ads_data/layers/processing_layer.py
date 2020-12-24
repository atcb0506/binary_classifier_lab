from typing import Any, Dict, List

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing \
    import Normalization, StringLookup


class CustomNormalizationLayer(Normalization):

    def __init__(
            self,
            feature_key: str,
            mean: np.ndarray = None,
            variance: np.ndarray = None,
    ) -> None:
        self.feature_key = feature_key
        super().__init__(
            name=f'{feature_key}_normalization_layer',
            mean=mean,
            variance=variance,
        )

    def adapt(self, data, reset_state=True) -> None:
        tf_autotune = tf.data.experimental.AUTOTUNE
        tmp_dataset = data.map(
            lambda feature, label: feature.get(self.feature_key),
            num_parallel_calls=tf_autotune,
            deterministic=False,
        )
        super().adapt(data=tmp_dataset, reset_state=reset_state)

    def get_config(self) -> Dict[str, Any]:
        weight = self.get_weights()
        config = {
            'feature_key': self.feature_key,
            'mean': weight[0],
            'variance': weight[1],
        }
        return config


class CustomStringLookupLayer(StringLookup):

    def __init__(
            self,
            feature_key: str,
            vocabulary: List[str] = None,
    ) -> None:

        self.feature_key = feature_key
        super().__init__(
            name=f'{feature_key}_stringlookup_layer',
            vocabulary=vocabulary
        )

    def adapt(self, data, reset_state=True) -> None:
        tf_autotune = tf.data.experimental.AUTOTUNE
        tmp_dataset = data.map(
            lambda feature, label: feature.get(self.feature_key),
            num_parallel_calls=tf_autotune,
            deterministic=False,
        )
        super().adapt(data=tmp_dataset, reset_state=reset_state)

    def get_config(self) -> Dict[str, Any]:
        vocabulary = self.get_vocabulary()
        vocabulary.remove('')
        vocabulary.remove('[UNK]')
        config = {
            'feature_key': self.feature_key,
            'vocabulary': vocabulary,
        }
        return config
