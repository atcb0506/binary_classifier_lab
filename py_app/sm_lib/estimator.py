from typing import Dict, List, Any

import sagemaker as sm
from sagemaker.estimator import Estimator
from sagemaker.tuner import HyperparameterTuner


class SagemakerTFEstimator(object):

    def __init__(
            self,
            container_image_uri: str,
            sm_session: sm.Session,
            sm_role: str,
            project_tag: List[Dict[str, str]],
            tn_instance_type: str,
            tn_instance_count: int,
            tn_volumesize: int,
            tn_job_name: str,
            max_run: int,
            shared_hyperparameters: Dict[str, str],
            **kwargs
    ) -> None:

        self.estimator = Estimator(
            image_uri=container_image_uri,
            instance_type=tn_instance_type,
            instance_count=tn_instance_count,
            volume_size=tn_volumesize,
            role=sm_role,
            sagemaker_session=sm_session,
            tags=project_tag,
            max_run=max_run,
            hyperparameters=shared_hyperparameters,
            **kwargs,
        )
        self._training_job_name = tn_job_name
        self._project_tag = project_tag

    def model_fit(
            self,
            inputs: Dict[str, str],
            hparam: Dict[str, Any] = None,
    ) -> None:

        if hparam is not None:

            tuner = HyperparameterTuner(
                estimator=self.estimator,
                objective_metric_name=hparam.get('objective_metric_name'),
                metric_definitions=hparam.get('metric_definitions'),
                hyperparameter_ranges=hparam.get('hyperparameter_ranges'),
                objective_type=hparam.get('objective_type'),
                max_jobs=hparam.get('max_jobs'),
                max_parallel_jobs=hparam.get('max_parallel_jobs'),
                tags=self._project_tag,
                base_tuning_job_name=self._training_job_name,
            )
            tuner.fit(
                inputs=inputs,
                job_name=self._training_job_name,
                wait=False,
                logs='All',
            )

        else:

            self.estimator.fit(
                inputs=inputs,
                job_name=self._training_job_name,
                wait=False,
                logs='All',
            )
