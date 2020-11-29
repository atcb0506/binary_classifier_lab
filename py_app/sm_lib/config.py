from typing import Any

import tensorflow as tf

from comm_lib.func import get_secret


class SagemakerPipelineConfig(object):

    def __init__(
            self,
            project_name: str,
            env: str,
            aws_region: str,
            current_time: str,
    ) -> None:

        self._ENV = env
        self._INSTANCE_TYPE = 'ml.c5.xlarge'  # 'ml.c5.4xlarge'
        self._INSTANCE_COUNT = 1
        self._PROJECT_NAME = project_name
        self._TAG = [
            {
                'Key': 'project',
                'Value': project_name
            },
            {
                'Key': 'env',
                'Value': env
            }
        ]
        self._PY_VERSION = 'py37'
        self._TF_VERSION = tf.__version__
        self._SECRET_NAME = f'{env}/sagemaker/config'
        self._REGION = aws_region
        self._MAX_RUN = 2 * 24 * 60 * 60
        self.current_time = current_time

    def getter(
            self,
            attr: str
    ) -> Any:

        sm_secret = get_secret(
            region_name=self._REGION,
            secret_name=f'{self._ENV}/sagemaker/config'
        )
        training_job_name = f'{self._PROJECT_NAME}-tn-{self.current_time}'

        data = {
            'sm_bucket': sm_secret['sm_bucket'],
            'sm_role': sm_secret['sm_role'],
            'sm_instance_type': self._INSTANCE_TYPE,
            'sm_instance_count': self._INSTANCE_COUNT,
            'project_tag': self._TAG,
            'training_job_name': training_job_name,
            'py_version': self._PY_VERSION,
            'tf_version': self._TF_VERSION,
            'max_run': self._MAX_RUN,
        }

        return data.get(attr)
