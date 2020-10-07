from typing import List, Dict, Any, Union, Tuple

import tensorflow as tf


class FeaturesExtraction:

    def __init__(
            self,
            num_col: List[str],
            feature_type: str = None
    ) -> None:

        self.num_col = num_col
        self.feature_type = feature_type

    def __call__(
            self,
            features: Dict[str, Any],
            labels: List[Any]
    ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], List[Any]]]:

        numeric_features = [features.pop(col) for col in self.num_col]
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)

        if self.feature_type == 'cat':
            return features
        if self.feature_type == 'numeric':
            return numeric_features
        if self.feature_type == 'no_label':
            features['numeric'] = numeric_features
            return features
        else:
            features['numeric'] = numeric_features
            return features, labels
