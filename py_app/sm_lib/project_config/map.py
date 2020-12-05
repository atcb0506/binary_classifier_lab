from typing import Type, Optional

from sm_lib.project_config.config import CriteoConfig


def get_project_config(
        project_name: str
) -> Optional[Type[CriteoConfig]]:

    config = {
        'criteo': CriteoConfig
    }

    return config.get(project_name)
