from abc import ABC, abstractmethod
from comm_lib.secret import get_secret


class BaseConfig(ABC):

    def __init__(
            self,
            project_name: str,
            env: str,
            region_name: str,
            current_time: str,
    ) -> None:
        self.sm_secret = get_secret(
            region_name=region_name,
            secret_name=f'{env}/sagemaker/config'
        )
        self.project_name = project_name
        self.current_time = current_time

    @abstractmethod
    def getter(self, attr: str):
        raise NotImplementedError

