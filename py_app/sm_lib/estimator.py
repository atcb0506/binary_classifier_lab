from typing import Dict, List

import sagemaker as sm
from sagemaker.tensorflow import TensorFlow


class SagemakerTFEstimator(TensorFlow):

    def __init__(
            self,
            sm_session: sm.Session,
            sm_role: str,
            tf_version: str,
            py_version: str,
            project_tag: List[Dict[str, str]],
            local_project_dir: str,
            tn_instance_type: str,
            tn_instance_count: int,
            tn_job_name: str,
            inputs: Dict[str, str],
            ep_instance_init_count: int,
            ep_instance_type: str,
            ep_name: str,
            **kwargs
    ) -> None:

        super().__init__(
            source_dir=local_project_dir,
            entry_point='run.py',
            instance_type=tn_instance_type,
            instance_count=tn_instance_count,
            role=sm_role,
            framework_version=tf_version,
            py_version=py_version,
            sagemaker_session=sm_session,
            tags=project_tag,
            **kwargs
        )
        self._input = inputs
        self._training_job_name = tn_job_name
        self._project_tag = project_tag
        self._ep_instance_init_count = ep_instance_init_count
        self._ep_instance_type = ep_instance_type
        self._ep_name = ep_name

    def fit(self, **kwargs) -> None:

        super().fit(
            inputs=self._input,
            job_name=self._training_job_name,
            wait=True,
            logs='All',
            **kwargs,
        )

    def deploy(self, **kwargs) -> None:

        _ = super().deploy(
            instance_type=self._ep_instance_type,
            initial_instance_count=self._ep_instance_init_count,
            endpoint_name=self._ep_name,
            tags=self._project_tag,
            model_name=self._ep_name,
        )
