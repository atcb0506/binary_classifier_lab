from datetime import datetime
import json
from typing import Any, Dict

import boto3
from botocore.exceptions import ClientError
import tensorflow as tf


class SagemakerPipelineConfig(object):

    def __init__(
            self,
            project_name: str,
            env: str) -> None:

        self._ENV = env
        self._INSTANCE_TYPE = 'ml.m5.large'
        self._INSTANCE_COUNT = 1
        self._EP_INIT_INSTANCE_TYPE = 'ml.m5.large'
        self._EP_INSTANCE_COUNT = 1
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
        self._REGION = 'ap-southeast-1'

    def getter(
            self,
            attr: str
    ) -> Any:

        sm_secret = self._get_secret(
            region_name=self._REGION,
            secret_name=f'{self._ENV}/sagemaker/config'
        )
        current_time = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        training_job_name = f'{self._PROJECT_NAME}-training-{current_time}'
        ep_name = f'{self._PROJECT_NAME}-endpoint-{current_time}'

        data = {
            'sm_bucket': sm_secret['sm_bucket'],
            'sm_role': sm_secret['sm_role'],
            'sm_instance_type': self._INSTANCE_TYPE,
            'sm_instance_count': self._INSTANCE_COUNT,
            'ep_instance_type': self._EP_INIT_INSTANCE_TYPE,
            'ep_instance_init_count': self._EP_INSTANCE_COUNT,
            'project_tag': self._TAG,
            'training_job_name': training_job_name,
            'ep_name': ep_name,
            'py_version': self._PY_VERSION,
            'tf_version': self._TF_VERSION,
        }

        return data.get(attr)

    @staticmethod
    def _get_secret(
            region_name: str,
            secret_name: str
    ) -> Dict[str, str]:

        session = boto3.session.Session()
        client = session.client(
            service_name='secretsmanager',
            region_name=region_name
        )

        try:
            get_secret_value_response = client.get_secret_value(
                SecretId=secret_name
            )
            return json.loads(get_secret_value_response['SecretString'])

        except ClientError as e:
            raise e
