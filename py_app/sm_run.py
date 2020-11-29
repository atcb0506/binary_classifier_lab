import argparse
from datetime import datetime

import sagemaker as sm

from comm_lib.metadata import ProjectMetaData
from sm_lib.config import SagemakerPipelineConfig
from sm_lib.estimator import SagemakerTFEstimator

if __name__ == '__main__':

    # args variable
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str)
    parser.add_argument('--env', type=str)
    parser.add_argument('--region', type=str)
    parser.add_argument('--mode', type=str, default='')
    args = parser.parse_args()
    project_name = args.project_name
    env = args.env
    mode = args.mode
    region = args.region
    current_time = datetime.utcnow().strftime('%Y%m%d%H%M%S')

    # config & metadata init
    config = SagemakerPipelineConfig(
        project_name=project_name,
        env=env,
        aws_region=region,
        current_time=current_time,
    )
    metadata = ProjectMetaData(
        project_name=project_name,
        env=env,
        current_time=current_time,
    ).getter()

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
        max_run=config.getter('max_run'),
        shared_hyperparameters=metadata.get('shared_hyperparameters'),
    )

    # fit
    estimator.model_fit(
        inputs=metadata.get('sm_input'),
        hparam=metadata.get('hparam_tuning'),
    )
