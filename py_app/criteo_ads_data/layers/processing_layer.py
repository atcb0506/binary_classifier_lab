import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing \
    import PreprocessingLayer, Normalization, CategoryEncoding, StringLookup


class DataProcessingLayer(PreprocessingLayer):

    def __init__(self, ls_cat_col, num_col, **kwargs):

        super(DataProcessingLayer, self).__init__(**kwargs)
        self._ls_cat_col = ls_cat_col
        self._num_col = num_col
        self._normalization_layer = Normalization(
            name=f'{num_col}_normalization_layer',
        )
        self._dict_stringlookup = dict()
        self._dict_categoryencodering = dict()
        for key in ls_cat_col:
            self._dict_stringlookup.update({
                key: StringLookup(name=f'{key}_indexer')
            })
            self._dict_categoryencodering.update({
                key: CategoryEncoding(
                    output_mode='binary',
                    name=f'{key}_encoder',
                )
            })

    def adapt(self, data, reset_state=True):

        self._adapt_normalizer(data=data)
        self._adapt_indexer(data=data)
        self._adapt_encoder(data=data)

    def _adapt_normalizer(self, data: tf.data.Dataset) -> None:

        print(f'adapting col: {self._num_col}')
        tmp_dataset = data.map(
            lambda feature, label: feature.get(self._num_col)
        )
        self._normalization_layer.adapt(tmp_dataset)

    def _adapt_indexer(self, data: tf.data.Dataset) -> None:

        for cat_col in self._ls_cat_col:
            tmp_dataset = data.map(lambda feature, label: feature.get(cat_col))
            indexer = self._dict_stringlookup[cat_col]
            print(f'adapting col: {cat_col} - indexer')
            indexer.adapt(tmp_dataset)

    def _adapt_encoder(self, data: tf.data.Dataset):

        for cat_col in self._ls_cat_col:
            tmp_dataset = data.map(lambda feature, label: feature.get(cat_col))
            indexer = self._dict_stringlookup[cat_col]
            indexed_dataset = tmp_dataset.map(lambda x: indexer(x))
            encoder = self._dict_categoryencodering[cat_col]
            print(f'adapting col: {cat_col} - encoder')
            encoder.adapt(indexed_dataset)

    def call(self, inputs, **kwargs):

        ls_processing_layers = list()
        normalization_layer = self._normalization_layer(inputs[self._num_col])
        ls_processing_layers.append(normalization_layer)
        for cat_col in self._ls_cat_col:
            indexer = self._dict_stringlookup[cat_col]
            encoder = self._dict_categoryencodering[cat_col]
            encoding_layer = encoder(indexer(inputs[cat_col]))
            ls_processing_layers.append(encoding_layer)

        return tf.keras.layers.concatenate(ls_processing_layers, axis=-1)
