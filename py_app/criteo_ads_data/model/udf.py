import os
import tensorflow as tf
from typing import Dict, Any
from tensorboard.plugins.hparams import api as hp

from model.embedding_model import EmbeddingModel


def model_fit(
        model: EmbeddingModel,
        training_set: tf.data.Dataset,
        validation_set: tf.data.Dataset,
        export_path: str,
        log_dir: str,
        hparams: Dict[hp.HParam, Any],
        epochs: int = 20,
        verbose: int = 1,
        worker: int = 4,
) -> None:

    # tensorboard logging for standard metrics
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        profile_batch=(300, 320)
    )

    # tensorboard logging for hyperparameters
    keras_callback = hp.KerasCallback(
        writer=log_dir,
        hparams=hparams,
        trial_id=log_dir
    )

    metrics = [
        tf.keras.metrics.AUC(name='auc'),
    ]
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=metrics)

    _ = model.fit(
        training_set,
        validation_data=validation_set,
        epochs=epochs,
        verbose=verbose,
        callbacks=[tensorboard_callback, keras_callback],
        workers=worker)

    model.summary()
    tf.saved_model.save(
        obj=model,
        export_dir=os.path.join(export_path, '1')
    )
