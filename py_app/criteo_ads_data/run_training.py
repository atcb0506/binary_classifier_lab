import argparse
import json
import logging
import os

from config import CAT_COLUMNS
from hparams.config import hparams_init
from dataprep.udf import train_test_prep
from model.embedding_model import EmbeddingModel
from model.udf import model_fit


def main(
        train_path: str,
        test_path: str,
        layer_dir: str,
        log_dir: str,
        export_dir: str,
        batch_size: int,
        embedding_dim_base: int,
        epochs: int,
) -> None:

    # init hparams
    hparams = hparams_init(
        epochs=epochs,
        batch_size=batch_size,
        log_dir=log_dir
    )
    lst_feature = CAT_COLUMNS
    lst_feature.append('numeric')

    # data preparation
    train, validate = train_test_prep(
        train_path=train_path,
        test_path=test_path,
        batch_size=batch_size
    )

    # model
    binary_classifier = EmbeddingModel(
        lst_features=lst_feature,
        layer_dir=layer_dir,
        name='binary_classifier',
        embedding_dim_base=embedding_dim_base,
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

    # init logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    # arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf_logs_path', type=str, default='../tensorboard')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--embedding_dim_base', type=int, default=10)
    parser.add_argument('--model_dir', type=str)
    args = parser.parse_args()

    # handle env
    train_path = os.environ.get('SM_CHANNEL_TRAIN')
    test_path = os.environ.get('SM_CHANNEL_TEST')
    export_dir = os.environ.get('SM_MODEL_DIR')
    layer_dir = os.environ.get('SM_CHANNEL_LAYER')
    job_name = json.loads(os.environ.get('SM_TRAINING_ENV'))['job_name']
    log_dir = f'{args.tf_logs_path}/log/{job_name}'

    # run program
    main(
        train_path=train_path,
        test_path=test_path,
        layer_dir=layer_dir,
        log_dir=log_dir,
        export_dir=export_dir,
        batch_size=args.batch_size,
        embedding_dim_base=args.embedding_dim_base,
        epochs=args.epochs,
    )
