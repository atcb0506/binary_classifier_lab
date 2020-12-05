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
        self.tag = [
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

    def getter(self, attr: str) -> Any:

        training_job_name = f'{self.project_name}-tn-{self.current_time}'
        data = {
            'sm_bucket': self.sm_secret['sm_bucket'],
            'sm_role': self.sm_secret['sm_role'],
            'sm_instance_type': 'ml.c5.2xlarge',
            'sm_instance_count': 1,
            'sm_volumesize': 100,
            'project_tag': self.tag,
            'training_job_name': training_job_name,
            'py_version': tf.__version__,
            'tf_version': 'py37',
            'max_run': 1 * 24 * 60 * 60,
        }

        return data.get(attr)


class SagemakerProcessingConfig(SagemakerBaseConfig):

    def getter(self, attr: str) -> Any:
        processing_job_name = f'{self.project_name}-pc-{self.current_time}'
        data = {
            'sm_bucket': self.sm_secret['sm_bucket'],
            'sm_role': self.sm_secret['sm_role'],
            'sm_instance_type': 'ml.c5.xlarge',
            'sm_instance_count': 20,
            'sm_volumesize': 100,
            'project_tag': self.tag,
            'processing_job_name': processing_job_name,
            'max_run': 1 * 24 * 60 * 60,
        }

        return data.get(attr)
