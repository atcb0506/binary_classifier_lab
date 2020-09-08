import os
import tensorflow as tf

from model.model import BinaryClassifier


def model_fit(
        model: BinaryClassifier,
        training_set: tf.data.Dataset,
        validation_set: tf.data.Dataset,
        export_path: str,
        epochs: int = 20,
        verbose: int = 1,
        worker: int = 4,
) -> None:

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
        workers=worker)

    model.summary()
    tf.saved_model.save(
        obj=model,
        export_dir=os.path.join(export_path, '1')
    )
