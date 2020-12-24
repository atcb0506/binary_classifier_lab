from abc import ABC, abstractmethod

from typing import Any
from sagemaker.tuner import IntegerParameter, CategoricalParameter
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.inputs import TrainingInput

from sm_lib.basic_config import BaseConfig


class ProjectBaseConfig(BaseConfig, ABC):

    def __init__(
            self,
            project_name: str,
            env: str,
            region_name: str,
            current_time: str,
    ) -> None:

        super().__init__(project_name, env, region_name, current_time)
        training_job_name = f'{self.project_name}-tn-{self.current_time}'
        self.tf_logs_path = f's3://{self.sm_secret["sm_bucket"]}' \
                            f'/{training_job_name}'

    @abstractmethod
    def getter(self, attr: str) -> Any:
        raise NotImplementedError


class CriteoConfig(ProjectBaseConfig):

    def getter(self, attr: str) -> Any:
        data = {
            'processing': {
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
                ]
            },
            'estimator': {
                'sm_input': {
                    'data_source': TrainingInput(
                        s3_data='s3://criteo-ads-data/prod/train_tfrecord_1000000_gz',
                        distribution='ShardedByS3Key',
                    )
                },
                'project_dir': 'criteo_ads_data',
                'shared_hyperparameters': {
                    'tf_logs_path': self.tf_logs_path,
                    'batch_size': 512,
                }
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
                        'Regex': '.*loss: [0-9\\.]+ - auc: ([0-9\\.]+).*'},
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
