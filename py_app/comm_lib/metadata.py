from typing import Dict, Any

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

        training_job_name = f'{self._PROJECT_NAME}-training-{self.current_time}'
        tf_logs_path = f's3://{self.sm_secret["sm_bucket"]}/{training_job_name}'

        data = {
            'criteo-ads-data': {
                'sm_input': {
                    'data_source': 's3://criteo-ads-data/prod/sample'
                },
                'project_dir': 'criteo_ads_data',
                'shared_hyperparameters': {
                    'data_filename': 'sample_train.txt',
                    'tf_logs_path': tf_logs_path,
                }
            }
        }

        return data.get(self._PROJECT_NAME)
