import os
from datetime import datetime
from datetime import timedelta

from airflow import DAG
from docker.types import Mount
from airflow.operators.docker_operator import DockerOperator



description = 'Este pipeline programa una secuencia de ejecuciones Docker'

default_args = {
    'owner'              : 'airflow',
    'schedule_interval'  : None,
    'description'        : description,
    'depend_on_past'     : False,
    'start_date'         : datetime.now(),
    'retries'            : 0,
    }

IMAGE = 'ml_project'
PROJECT_DIR= os.environ['PROJECT_DIR']

params = {
    'image_name'          : 'my_etl',
    'param_max_iter'      : 100
}

with DAG(
    'DataOps',
    default_args=default_args,
    params=params,
    catchup=False,
) as dag:

    brz = DockerOperator(
        task_id='bronze_step',
        image="{{ params.image_name }}",
        auto_remove=True,
        network_mode="01_datalakehouse_default",
        entrypoint=['python','brz/origin_1.py'],
        environment={
            "PARAM_MAX_ITER" : "{{ params.param_max_iter }}",
        },
    )
    slv = DockerOperator(
        task_id='silver_step',
        image="{{ params.image_name }}",
        network_mode="01_datalakehouse_default",
        auto_remove=True,
        entrypoint=['python','slv/app_real_state.py'],
        environment={
            "PARAM_MAX_ITER" : "{{ params.param_max_iter }}",
        },
    )

    gld = DockerOperator(
        task_id='gold_step',
        image="{{ params.image_name }}",
        network_mode="01_datalakehouse_default",
        auto_remove=True,
        entrypoint=['python','gld/app_real_state.py'],
        environment={
            "PARAM_MAX_ITER" : "{{ params.param_max_iter }}",
        },
    )


    brz >> slv >> gld
