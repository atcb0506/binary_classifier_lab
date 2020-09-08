from typing import List

import tensorflow as tf


class ANNLayer(tf.keras.layers.Layer):

    def __init__(
            self,
            num_hidden_layer: int,
            ls_hidden_unit: List[int],
            **kwargs
    ) -> None:

        super(ANNLayer, self).__init__(**kwargs)

        assert num_hidden_layer == len(ls_hidden_unit), f'num_hidden_layer != len(ls_hidden_unit)'
        self.num_hidden_layer = num_hidden_layer

        # fully connected hidden layers
        self.ls_hidden_layer = list()
        for i in range(num_hidden_layer):
            self.ls_hidden_layer.append(
                tf.keras.layers.Dense(
                    ls_hidden_unit[i],
                    activation='relu',
                    name=f'hidden_layer_{i}')
            )

    def call(self, inputs, **kwargs):  # pylint: disable=unused-argument

        layer = inputs
        for i in range(self.num_hidden_layer):
            layer = self.ls_hidden_layer[i](layer)

        return layer
