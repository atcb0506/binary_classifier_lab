import argparse
from datetime import datetime

import sagemaker as sm

from sm_lib.config.map import get_project_config
from sm_lib.config.sm_config import SagemakerTrainingConfig
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
    sm_config = SagemakerTrainingConfig(
        project_name=project_name,
        env=env,
        region_name=region,
        current_time=current_time,
    )
    metadata = get_project_config(project_name)(
        project_name=project_name,
        env=env,
        region_name=region,
        current_time=current_time,
    )
    estimator_config = metadata.getter('estimator')
    hparam_config = None
    if mode == 'hparams':
        hparam_config = metadata.getter('hparam_tuning')

    # create sagemaker session
    sess = sm.Session(
        default_bucket=sm_config.getter('sm_bucket')
    )

    # create estimator
    estimator = SagemakerTFEstimator(
        sm_session=sess,
        sm_role=sm_config.getter('sm_role'),
        tf_version=sm_config.getter('tf_version'),
        py_version=sm_config.getter('py_version'),
        project_tag=sm_config.getter('project_tag'),
        local_project_dir=estimator_config.get('project_dir'),
        tn_instance_type=sm_config.getter('sm_instance_type'),
        tn_instance_count=sm_config.getter('sm_instance_count'),
        tn_volumesize=sm_config.getter('sm_volumesize'),
        tn_job_name=sm_config.getter('training_job_name'),
        max_run=sm_config.getter('max_run'),
        shared_hyperparameters=estimator_config.get('shared_hyperparameters'),
    )

    # fit
    estimator.model_fit(
        inputs=estimator_config.get('sm_input'),
        hparam=hparam_config,
    )
