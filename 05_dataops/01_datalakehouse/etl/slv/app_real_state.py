import os
import io
import boto3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def app_real_state():


    process_date="202406"

    source = boto3.client('s3',
        endpoint_url = os.environ["AWS_S3_URI"],
        aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"],
    ).get_object(
        Bucket='brz',
        Key=f'origen_1/{process_date}/housing.csv',
    )

    tarjet = io.BytesIO()

    pd.read_csv(
        source["Body"],
    ).to_parquet(
        tarjet,
        index=False,
        engine='pyarrow',
    )

    tarjet.seek(0)

    boto3.client('s3',
        endpoint_url = os.environ["AWS_S3_URI"],
        aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"],
    ).upload_fileobj(
        tarjet,
        'slv',
        f'app_real_state/housing.parquet',
    )

if __name__=='__main__':

    app_real_state()
