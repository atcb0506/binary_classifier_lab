import argparse
import os

from typing import Dict

from loader.udf import dataprep
from layers.udf import build_data_processing_layer
from layers.ann_layer import ANNLayer
from model.model import BinaryClassifier
from model.udf import model_fit


def main(
        data_path: str,
        data_filename: str,
        tf_logs_path: str,
        export_dir: str,
        batch_size: int,
        epochs: int,
) -> None:

    # data preparation
    train, validate = dataprep(
        data_path=os.path.join(data_path, data_filename),
        batch_size=batch_size
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
        export_path=export_dir,
        tf_logs_path=tf_logs_path,
        epochs=epochs,
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        default=os.environ.get('SM_CHANNEL_DATA_SOURCE'),
    )
    parser.add_argument(
        '--export_dir',
        type=str,
        default=os.environ.get('SM_MODEL_DIR'),
    )
    parser.add_argument('--data_filename', type=str, default='sample_train.txt')
    parser.add_argument('--tf_logs_path', type=str, default='../saved')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--model_dir', type=str)
    args = parser.parse_args()

    main(
        data_path=args.data_path,
        data_filename=args.data_filename,
        tf_logs_path=args.tf_logs_path,
        export_dir=args.export_dir,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
