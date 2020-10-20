import argparse
import os

import sagemaker as sm

from comm_lib.metadata import data
from sm_lib.config import SagemakerPipelineConfig
from sm_lib.estimator import SagemakerTFEstimator

if __name__ == '__main__':

    # args variable
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str)
    parser.add_argument('--env', type=str)
    parser.add_argument('--mode', type=str, default='')
    args = parser.parse_args()
    project_name = args.project_name
    env = args.env
    mode = args.mode

    # config & metadata init
    config = SagemakerPipelineConfig(
        project_name=project_name,
        env=env
    )
    metadata = data.get(project_name)

    # create sagemaker session
    sess = sm.Session(
        default_bucket=config.getter('sm_bucket')
    )

    # create estimator
    estimator = SagemakerTFEstimator(
        sm_session=sess,
        sm_role=config.getter('sm_role'),
        tf_version=config.getter('tf_version'),
        py_version=config.getter('py_version'),
        project_tag=config.getter('project_tag'),
        local_project_dir=metadata.get('project_dir'),
        tn_instance_type=config.getter('sm_instance_type'),
        tn_instance_count=config.getter('sm_instance_count'),
        tn_job_name=config.getter('training_job_name'),
        inputs=metadata.get('sm_input'),
        ep_instance_init_count=config.getter('ep_instance_init_count'),
        ep_instance_type=config.getter('ep_instance_type'),
        ep_name=config.getter('ep_name'),
    )

    # fit
    estimator.fit()

    # deploy
    if mode == 'deploy':
        estimator.deploy()
