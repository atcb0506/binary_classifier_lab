import argparse
from datetime import datetime

import sagemaker as sm
from sagemaker.processing import Processor

from sm_lib.sm_config.config import SagemakerProcessingConfig
from sm_lib.project_config.map import get_project_config

if __name__ == '__main__':

    # args variable
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str)
    parser.add_argument('--env', type=str)
    parser.add_argument('--region', type=str)
    parser.add_argument('--image_uri', type=str)
    args = parser.parse_args()
    project_name = args.project_name
    env = args.env
    image_uri = args.image_uri
    region = args.region
    current_time = datetime.utcnow().strftime('%Y%m%d%H%M%S')

    # config & metadata init
    sm_config = SagemakerProcessingConfig(
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
    proc_config = metadata.getter('processing')

    # create sagemaker session
    sess = sm.Session(
        default_bucket=sm_config.getter('sm_bucket')
    )

    processor = Processor(
        role=sm_config.getter('sm_role'),
        image_uri=image_uri,
        instance_count=sm_config.getter('sm_instance_count'),
        instance_type=sm_config.getter('sm_instance_type'),
        entrypoint=proc_config.get('endpoint'),
        volume_size_in_gb=sm_config.getter('sm_volumesize'),
        sagemaker_session=sess,
        tags=sm_config.getter('project_tag'),
    )

    processor.run(
        inputs=proc_config.get('inputs'),
        outputs=proc_config.get('outputs'),
        arguments=proc_config.get('arguments'),
        wait=False,
        logs=False,
        job_name=sm_config.getter('processing_job_name'),
    )
