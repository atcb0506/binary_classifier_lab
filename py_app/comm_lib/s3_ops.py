import os
import logging

import boto3


def download_dir(
        bucket: str,
        prefix: str,
        local_logdir: str,
) -> None:

    keys = []
    dirs = []
    next_token = ''
    base_kwargs = {
        'Bucket': bucket,
        'Prefix': prefix,
    }

    client = boto3.client('s3')
    while next_token is not None:
        kwargs = base_kwargs.copy()
        if next_token != '':
            kwargs.update({'ContinuationToken': next_token})
        results = client.list_objects_v2(**kwargs)
        contents = results.get('Contents')
        for i in contents:
            k = i.get('Key')
            if k[-1] != '/':
                keys.append(k)
            else:
                dirs.append(k)
        next_token = results.get('NextContinuationToken')
    for d in dirs:
        dest_pathname = os.path.join(local_logdir, d)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            logging.info(f'Creating directory {dest_pathname}...')
            os.makedirs(os.path.dirname(dest_pathname))
    for k in keys:
        dest_pathname = os.path.join(local_logdir, k)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
        logging.info(f'Downloading {k}...')
        client.download_file(bucket, k, dest_pathname)
