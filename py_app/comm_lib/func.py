import json
from typing import Dict

import boto3
from botocore.exceptions import ClientError


def get_secret(
        region_name: str,
        secret_name: str
) -> Dict[str, str]:
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
        return json.loads(get_secret_value_response['SecretString'])

    except ClientError as e:
        raise e
