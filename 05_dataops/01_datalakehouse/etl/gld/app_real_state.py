import os
import io
import boto3
import sqlalchemy
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def app_real_state():


    source = boto3.client('s3',
        endpoint_url = os.environ["AWS_S3_URI"],
        aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"],
    ).get_object(
        Bucket='slv',
        Key=f'app_real_state/housing.parquet',
    )

    df = pd.read_parquet(io.BytesIO(source['Body'].read()))

    df.to_sql(
        'housing_data',
        sqlalchemy.create_engine(os.environ["DB_CONNECTION_STRING"]),
        if_exists='replace',
        index=False,
    )


if __name__=='__main__':

    app_real_state()
