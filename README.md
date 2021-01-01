# Binary classifier lab

This project is to test case the end2end ML approach on AWS Sagemaker

---

## 1. Training jobs

|Project|Data source|Project Directory|
|:---|:---|:---|
|Criteo ads data|[Criteo ads data link]|[Criteo project dir]|

[Criteo ads data link]: http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/
[Criteo project dir]: py_app/criteo_ads_data

## 2. Sagemaker processing - .tfrecord conversion

```shell script
python sagemaker_jobs/run_sm_processing.py
--env=test
--project_name=criteo
--region=ap-southeast-1
--image_uri=558467021483.dkr.ecr.ap-southeast-1.amazonaws.com/binary_classifier_lab:4.3.3-PROCSSING
--prcossing_task=tfrecord_processing
```

## 3. Sagemaker processing - preprocessing layer

```shell script
python sagemaker_jobs/run_sm_processing.py
--env=test
--project_name=criteo
--region=ap-southeast-1
--image_uri=558467021483.dkr.ecr.ap-southeast-1.amazonaws.com/binary_classifier_lab:4.3.3-PROCSSING
--prcossing_task=layer_processing
```

## 4. Sagemaker training - no hyperparameter tuning

```shell script
python sagemaker_jobs/run_sm_training.py \
--env=test
--project_name=criteo
--region=ap-southeast-1
--container_image_uri=558467021483.dkr.ecr.ap-southeast-1.amazonaws.com/binary_classifier_lab:4.3.3-TRAINING
```

## 5. Sagemaker training - with hyperparameter tuning

```shell script
python sagemaker_jobs/run_sm_training.py \
--env=test
--project_name=criteo
--region=ap-southeast-1
--container_image_uri=558467021483.dkr.ecr.ap-southeast-1.amazonaws.com/binary_classifier_lab:4.3.3-TRAINING
--mode=hparams
```