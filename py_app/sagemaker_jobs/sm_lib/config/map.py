from typing import Type, Optional

from sagemaker_jobs.sm_lib.config.proj_config import CriteoConfig


def get_project_config(
        project_name: str
) -> Optional[Type[CriteoConfig]]:

    config = {
        'criteo': CriteoConfig
    }

    return config.get(project_name)
