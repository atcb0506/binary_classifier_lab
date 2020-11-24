from typing import Dict, Any
from sagemaker.tuner import IntegerParameter, CategoricalParameter

from comm_lib.func import get_secret


class ProjectMetaData(object):

    def __init__(
            self,
            project_name: str,
            env: str,
            current_time: str,
    ) -> None:

        self.sm_secret = get_secret(
            region_name='ap-southeast-1',
            secret_name=f'{env}/sagemaker/config'
        )
        self._PROJECT_NAME = project_name
        self.current_time = current_time

    def getter(self) -> Dict[str, Any]:

        training_job_name = f'{self._PROJECT_NAME}-tn-{self.current_time}'
        tf_logs_path = f's3://{self.sm_secret["sm_bucket"]}/{training_job_name}'

        data = {
            'criteo': {
                'sm_input': {
                    'data_source': 's3://criteo-ads-data/prod/sample'
                },
                'project_dir': 'criteo_ads_data',
                'shared_hyperparameters': {
                    'data_filename': 'sample_train.txt',
                    'tf_logs_path': tf_logs_path,
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
                }
            }
        }

        return data.get(self._PROJECT_NAME)
