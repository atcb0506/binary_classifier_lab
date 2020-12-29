from abc import ABC, abstractmethod
from typing import Any

import tensorflow as tf

from sm_lib.config.basic_config import BaseConfig


class SagemakerBaseConfig(BaseConfig, ABC):

    def __init__(
            self,
            **kwargs
    ) -> None:

        super().__init__(**kwargs)
        self.tag = [
            {
                'Key': 'project',
                'Value': self.project_name
            },
            {
                'Key': 'env',
                'Value': self.env
            }
        ]

    @abstractmethod
    def getter(self, attr: str) -> Any:
        raise NotImplementedError


class SagemakerTrainingConfig(SagemakerBaseConfig):

    def __init__(
            self,
            sm_instance_type: str,
            sm_instance_count: int,
            sm_volumesize: int,
            max_run: int,
            **kwargs,
    ) -> None:

        super().__init__(**kwargs)
        self.sm_instance_type = sm_instance_type
        self.sm_instance_count = sm_instance_count
        self.sm_volumesize = sm_volumesize
        self.max_run = max_run

    def getter(self, attr: str) -> Any:

        training_job_name = f'{self.project_name}-tn-{self.current_time}'
        data = {
            'sm_bucket': self.sm_secret['sm_bucket'],
            'sm_role': self.sm_secret['sm_role'],
            'sm_instance_type': self.sm_instance_type,
            'sm_instance_count': self.sm_instance_count,
            'sm_volumesize': self.sm_volumesize,
            'project_tag': self.tag,
            'training_job_name': training_job_name,
            'max_run': self.max_run,
        }

        return data.get(attr)


class SagemakerProcessingConfig(SagemakerBaseConfig):

    def __init__(
            self,
            sm_instance_type: str,
            sm_instance_count: int,
            sm_volumesize: int,
            max_run: int,
            **kwargs
    ) -> None:

        super().__init__(**kwargs)
        self.sm_instance_type = sm_instance_type
        self.sm_instance_count = sm_instance_count
        self.sm_volumesize = sm_volumesize
        self.max_run = max_run

    def getter(self, attr: str) -> Any:
        processing_job_name = f'{self.project_name}-pc-{self.current_time}'
        data = {
            'sm_bucket': self.sm_secret['sm_bucket'],
            'sm_role': self.sm_secret['sm_role'],
            'sm_instance_type': self.sm_instance_type,
            'sm_instance_count': self.sm_instance_count,
            'sm_volumesize': self.sm_volumesize,
            'project_tag': self.tag,
            'processing_job_name': processing_job_name,
            'max_run': self.max_run,
        }

        return data.get(attr)
