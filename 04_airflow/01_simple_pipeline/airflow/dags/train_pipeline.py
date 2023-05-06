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

PROJECT_DIR ='/home/adelgado/Documentos/geekshub_mlops/04_airflow/01_simple_pipeline'
IMAGE = 'ml_project'

params = {
    'image_name'          : IMAGE,
    'dataset_name'        : 'iris.csv',
    'model_name'          : IMAGE + '.pkl',
    'param_max_iter'      : 100,
}

with DAG(
    'TrainingPipeline',
    default_args=default_args,
    params=params,
    catchup=False,
) as dag:



    t1 = DockerOperator(
        task_id='training_step',
        image="{{ params.image_name }}",
        auto_remove=True,
        entrypoint=['sh','train.sh'],
        mount_tmp_dir=True,
        mounts=[
            Mount(
                source = f'{PROJECT_DIR}/data',
                target='/home/input/',
                type='bind'
            ),
            Mount(
                source = f'{PROJECT_DIR}/model/',
                target='/home/output/',
                type='bind'
            )
        ],
        environment={
            "PARAM_MAX_ITER" : "{{ params.param_max_iter }}",
            "DATA_PATH"      : "/home/input/{{ params.dataset_name }}",
            "MODEL_PATH"     : "/home/output/{{ params.model_name }}"
        },
    )


    t1
