from abc import ABC, abstractmethod

from typing import Any, Dict
from sagemaker.tuner import IntegerParameter, CategoricalParameter
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.inputs import TrainingInput

from sagemaker_jobs.config.basic_config import BaseConfig
from sagemaker_jobs.config.sm_config import \
    SagemakerProcessingConfig, SagemakerTrainingConfig


class ProjectBaseConfig(BaseConfig, ABC):

    def __init__(
            self,
            **kwargs,
    ) -> None:

        super().__init__(**kwargs)
        training_job_name = f'{self.project_name}-tn-{self.current_time}'
        self.tf_logs_path = f's3://{self.sm_secret["sm_bucket"]}' \
                            f'/{training_job_name}'

    @abstractmethod
    def getter(self, attr: str) -> Any:
        raise NotImplementedError


class CriteoConfig(ProjectBaseConfig):

    def getter(self, attr: str) -> Dict[str, Any]:
        data = {
            'tfrecord_processing': {
                'endpoint': ['python3', 'criteo_ads_data/run_processing.py'],
                'inputs': [
                    ProcessingInput(
                        source='s3://criteo-ads-data/prod/train_csv',
                        destination='/opt/ml/processing/input',
                        s3_data_distribution_type='ShardedByS3Key',
                    )
                ],
                'outputs': [
                    ProcessingOutput(
                        source='/opt/ml/processing/output',
                        destination='s3://criteo-ads-data/prod/train_tfrecord_gz',
                    )
                ],
                'arguments': [
                    '--input_path=/opt/ml/processing/input',
                    '--output_path=/opt/ml/processing/output',
                ],
                'sm_config': SagemakerProcessingConfig(
                    project_name=self.project_name,
                    env=self.env,
                    region_name=self.region_name,
                    current_time=self.current_time,
                    sm_instance_type='ml.c5.2xlarge',
                    sm_instance_count=20,
                    sm_volumesize=100,
                    max_run=1 * 60 * 60,
                )
            },
            'layer_processing': {
                'endpoint': ['python3', 'criteo_ads_data/run_processing_layer.py'],
                'inputs': [
                    ProcessingInput(
                        source='s3://criteo-ads-data/prod/train_tfrecord_gz/train',
                        destination='/opt/ml/processing/input',
                        s3_data_distribution_type='FullyReplicated',
                    )
                ],
                'outputs': [
                    ProcessingOutput(
                        source='/opt/ml/processing/output',
                        destination='s3://criteo-ads-data/prod/proc_layer',
                    )
                ],
                'arguments': [
                    '--input_path=/opt/ml/processing/input',
                    '--output_path=/opt/ml/processing/output',
                ],
                'sm_config': SagemakerProcessingConfig(
                    project_name=self.project_name,
                    env=self.env,
                    region_name=self.region_name,
                    current_time=self.current_time,
                    sm_instance_type='ml.c5.9xlarge',
                    sm_instance_count=1,
                    sm_volumesize=100,
                    max_run=24 * 60 * 60,
                )
            },
            'estimator': {
                'sm_input': {
                    'train': TrainingInput(
                        s3_data='s3://criteo-ads-data/prod/train_tfrecord_100000_gz/train',
                        distribution='FullyReplicated',
                    ),
                    'test': TrainingInput(
                        s3_data='s3://criteo-ads-data/prod/train_tfrecord_100000_gz/test',
                        distribution='FullyReplicated',
                    ),
                    'layer': TrainingInput(
                        s3_data='s3://criteo-ads-data/prod/proc_layer_100000',
                        distribution='FullyReplicated',
                    ),
                },
                'shared_hyperparameters': {
                    'tf_logs_path': self.tf_logs_path,
                    'batch_size': 512,
                },
                'sm_config': SagemakerTrainingConfig(
                    project_name=self.project_name,
                    env=self.env,
                    region_name=self.region_name,
                    current_time=self.current_time,
                    sm_instance_type='ml.c5.2xlarge',
                    sm_instance_count=1,
                    sm_volumesize=300,
                    max_run=1 * 24 * 60 * 60,
                )
            },
            'hparam_tuning': {
                'objective_metric_name': 'validation:loss',
                'metric_definitions': [
                    {
                        'Name': 'train:loss',
                        'Regex': '.*loss: ([0-9\\.]+) - auc: [0-9\\.]+.*'
                    },
                    {
                        'Name': 'train:auc',
                        'Regex': '.*loss: [0-9\\.]+ - auc: ([0-9\\.]+).*'
                    },
                    {
                        'Name': 'validation:loss',
                        'Regex': '.*step - loss: [0-9\\.]+ - auc: [0-9\\.]+ - val_loss: ([0-9\\.]+) - val_auc: [0-9\\.]+.*'
                    },
                    {
                        'Name': 'validation:auc',
                        'Regex': '.*step - loss: [0-9\\.]+ - auc: [0-9\\.]+ - val_loss: [0-9\\.]+ - val_auc: ([0-9\\.]+).*'
                    },
                ],
                'hyperparameter_ranges': {
                    'epochs': IntegerParameter(1, 50),
                    'batch_size': CategoricalParameter([64, 128, 256, 512])
                },
                'objective_type': 'Minimize',
                'max_jobs': 5,
                'max_parallel_jobs': 5,
            },
        }

        return data.get(attr)
