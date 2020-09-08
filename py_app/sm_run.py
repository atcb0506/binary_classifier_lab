import os

import sagemaker as sm

from comm_lib.metadata import data
from sm_lib.config import SagemakerPipelineConfig
from sm_lib.model_training import sagemaker_training

if __name__ == '__main__':

    # init
    project_name = os.environ['project_name']
    env = os.environ['env']
    config = SagemakerPipelineConfig(
        project_name=project_name,
        env=env
    )
    metadata = data.get(project_name)

    # create sagemaker session
    sess = sm.Session(
        default_bucket=config.getter('sm_bucket')
    )

    # training
    sagemaker_training(
        session=sess,
        sm_instance_type=config.getter('sm_instance_type'),
        sm_instance_count=config.getter('sm_instance_count'),
        sm_role=config.getter('sm_role'),
        tf_version=config.getter('tf_version'),
        py_version=config.getter('py_version'),
        project_tag=config.getter('project_tag'),
        project_dir=metadata.get('project_dir'),
        training_job_name=config.getter('training_job_name'),
        inputs=metadata.get('sm_input'),
    )
