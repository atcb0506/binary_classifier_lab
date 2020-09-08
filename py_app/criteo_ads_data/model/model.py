import tensorflow as tf

from layers.ann_layer import ANNLayer
from layers.processing_layer import DataProcessingLayer


class BinaryClassifier(tf.keras.Model):

    def __init__(
            self,
            processing_layer: DataProcessingLayer,
            model_architecture: ANNLayer,
            **kwargs
    ) -> None:

        super(BinaryClassifier, self).__init__(**kwargs)
        self.processing_layer = processing_layer
        self.model_architecture = model_architecture
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')

    def call(
            self,
            inputs,
            training=None,
            mask=None
    ):

        block1 = self.processing_layer(inputs)
        block2 = self.model_architecture(block1)
        output = self.output_layer(block2)

        return output
