import os
import io
import boto3
from dotenv import load_dotenv

load_dotenv()



def origin_2():

    source = io.BytesIO()
    boto3.client('s3',
        endpoint_url = os.environ["AWS_S3_URI"],
        aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"],
    ).download_fileobj(
        'stg',
        'origen_2/housing.json',
        source)

    target = source
    process_date="202406"
    target.seek(0)

    boto3.client('s3',
        endpoint_url = os.environ["AWS_S3_URI"],
        aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"],
    ).upload_fileobj(
        target,
        'brz',
        f'origen_2/{process_date}/housing.json',
    )


if __name__=="__main__":

    origin_2()
