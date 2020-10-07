from typing import Dict

import sagemaker as sm
from sagemaker.tensorflow import TensorFlow


def sagemaker_training(
        session: sm.Session,
        sm_instance_type: str,
        sm_instance_count: int,
        sm_role: str,
        tf_version: str,
        py_version: str,
        project_tag: str,
        project_dir: str,
        training_job_name: str,
        inputs: Dict[str, str]
) -> None:

    estimator = TensorFlow(
        source_dir=project_dir,
        entry_point='run.py',
        instance_type=sm_instance_type,
        instance_count=sm_instance_count,
        role=sm_role,
        framework_version=tf_version,
        py_version=py_version,
        sagemaker_session=session,
        tags=project_tag
    )

    estimator.fit(
        inputs=inputs,
        job_name=training_job_name,
        wait=True,
        logs='All'
    )
