from abc import ABC, abstractmethod
from typing import Any

import tensorflow as tf

from sm_lib.basic_config import BaseConfig


class SagemakerBaseConfig(BaseConfig, ABC):

    def __init__(
            self,
            project_name: str,
            env: str,
            region_name: str,
            current_time: str,
    ) -> None:

        super().__init__(project_name, env, region_name, current_time)
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

    @abstractmethod
    def getter(self, attr: str) -> Any:
        raise NotImplementedError


class SagemakerTrainingConfig(SagemakerBaseConfig):

    def __init__(
            self,
            project_name: str,
            env: str,
            region_name: str,
            current_time: str,
    ) -> None:

        super().__init__(project_name, env, region_name, current_time)
        self._INSTANCE_TYPE = 'ml.c5.2xlarge'
        self._INSTANCE_COUNT = 1
        self._VOLUMESIZE = 100
        self._MAX_RUN = 1 * 24 * 60 * 60
        self._PY_VERSION = 'py37'
        self._TF_VERSION = tf.__version__

    def getter(self, attr: str) -> Any:

        training_job_name = f'{self.project_name}-tn-{self.current_time}'
        data = {
            'sm_bucket': self.sm_secret['sm_bucket'],
            'sm_role': self.sm_secret['sm_role'],
            'sm_instance_type': self._INSTANCE_TYPE,
            'sm_instance_count': self._INSTANCE_COUNT,
            'sm_volumesize': self._VOLUMESIZE,
            'project_tag': self._TAG,
            'training_job_name': training_job_name,
            'py_version': self._PY_VERSION,
            'tf_version': self._TF_VERSION,
            'max_run': self._MAX_RUN,
        }

        return data.get(attr)


class SagemakerProcessingConfig(SagemakerBaseConfig):

    def __init__(
            self,
            project_name: str,
            env: str,
            region_name: str,
            current_time: str,
    ) -> None:
        super().__init__(project_name, env, region_name, current_time)
        self._INSTANCE_TYPE = 'ml.c5.2xlarge'
        self._INSTANCE_COUNT = 1
        self._VOLUMESIZE = 100
        self._MAX_RUN = 1 * 24 * 60 * 60

    def getter(self, attr: str) -> Any:
        processing_job_name = f'{self.project_name}-pc-{self.current_time}'
        data = {
            'sm_bucket': self.sm_secret['sm_bucket'],
            'sm_role': self.sm_secret['sm_role'],
            'sm_instance_type': self._INSTANCE_TYPE,
            'sm_instance_count': self._INSTANCE_COUNT,
            'sm_volumesize': self._VOLUMESIZE,
            'project_tag': self._TAG,
            'processing_job_name': processing_job_name,
            'max_run': self._MAX_RUN,
        }

        return data.get(attr)
