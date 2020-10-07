import argparse
import os

from typing import Dict

from loader.udf import dataprep
from layers.udf import build_data_processing_layer
from layers.ann_layer import ANNLayer
from model.model import BinaryClassifier
from model.udf import model_fit


def _parsed_args() -> Dict[str, str]:

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source', type=str, default=os.environ.get('SM_CHANNEL_DATA_SOURCE'))
    parser.add_argument('--export_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--model_dir', type=str)  # TODO: not sure the exact use case

    return vars(parser.parse_args())


def main(
        data_source: str,
        export_dir: str,
        model_dir: str,
) -> None:

    # data preparation
    train, validate = dataprep(
        data_source=os.path.join(data_source, 'sample_test.txt'),
        batch_size=200
    )

    # data precessing
    data_processing_layer = build_data_processing_layer(
        dataset=train
    )

    # ann layer
    ann_block_layer = ANNLayer(
        num_hidden_layer=2,
        ls_hidden_unit=[128, 128],
        name='fully_connected_layer'
    )

    # model
    binary_classifier = BinaryClassifier(
        processing_layer=data_processing_layer,
        model_architecture=ann_block_layer,
        name='binary_classifier'
    )

    # training
    model_fit(
        model=binary_classifier,
        training_set=train,
        validation_set=validate,
        export_path=export_dir
    )


if __name__ == '__main__':

    main(**_parsed_args())
