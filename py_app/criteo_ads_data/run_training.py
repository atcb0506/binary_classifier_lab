import argparse
import json
import os

from hparams.config import hparams_init
from loader.udf import dataprep
from layers.udf import build_data_processing_layer
from layers.ann_layer import ANNLayer
from model.model import BinaryClassifier
from model.udf import model_fit


def main(
        data_path: str,
        data_filename: str,
        log_dir: str,
        export_dir: str,
        batch_size: int,
        epochs: int,
) -> None:

    # init hparams
    hparams = hparams_init(
        epochs=epochs,
        batch_size=batch_size,
        log_dir=log_dir
    )

    # data preparation
    train, validate = dataprep(
        data_path=data_path,
        data_filename=data_filename,
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
        log_dir=log_dir,
        hparams=hparams,
        epochs=epochs,
    )


if __name__ == '__main__':

    # arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_filename', type=str,
                        default='tfrecord_10000.tfrecord')
    parser.add_argument('--tf_logs_path', type=str, default='../tensorboard')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--model_dir', type=str)
    args = parser.parse_args()

    # handle env
    data_path = os.environ.get('SM_CHANNEL_DATA_SOURCE')
    export_dir = os.environ.get('SM_MODEL_DIR')
    job_name = json.loads(os.environ.get('SM_TRAINING_ENV'))['job_name']
    log_dir = f'{args.tf_logs_path}/log/{job_name}'

    # run program
    main(
        data_path=data_path,
        data_filename=args.data_filename,
        log_dir=log_dir,
        export_dir=export_dir,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
