import tensorflow as tf
from typing import Dict, Any
from tensorboard.plugins.hparams import api as hp

HP_EPOCHS = hp.HParam('epochs', hp.IntInterval(1, 50))
HP_BATCH_SIZE = hp.HParam(
    'batch_size',
    hp.Discrete([64, 128, 256, 512, 1024, 2048])
)
HPARAMS = [HP_EPOCHS, HP_BATCH_SIZE]
METRICS = [
    hp.Metric('epoch_loss', group='train',
              display_name='epoch_loss (train)'),
    hp.Metric('epoch_loss', group='validation',
              display_name='epoch_loss (validation)'),
    hp.Metric('epoch_auc', group='train',
              display_name='epoch_auc (train)'),
    hp.Metric('epoch_auc', group='validation',
              display_name='epoch_auc (validation)'),
]


def hparams_init(
        epochs: int,
        batch_size: int,
        log_dir: str,
) -> Dict[hp.HParam, Any]:

    # log directories
    with tf.summary.create_file_writer(log_dir).as_default():
        hp.hparams_config(hparams=HPARAMS, metrics=METRICS)

    # hyperparameter setting
    hparams = {
        HP_EPOCHS: epochs,
        HP_BATCH_SIZE: batch_size,
    }

    return hparams
