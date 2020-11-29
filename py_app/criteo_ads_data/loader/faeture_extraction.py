from typing import List, Dict, Any, Union, Tuple

import tensorflow as tf


class FeaturesExtraction:

    def __init__(
            self,
            num_col: List[str],
            lbl_col: str,
    ) -> None:

        self.num_col = num_col
        self.lbl_col = lbl_col

    def __call__(
            self,
            features: Dict[str, Any]
    ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], List[Any]]]:

        numeric_features = [features.pop(col) for col in self.num_col]
        numeric_features = [
            tf.cast(feat, tf.float32) for feat in numeric_features
        ]
        numeric_features = tf.stack(numeric_features, axis=-1)

        features['numeric'] = numeric_features
        return features, features[self.lbl_col]
