import os
from datetime import datetime
from datetime import timedelta

from airflow import DAG
from docker.types import Mount
from plugins.operators import DockerOperatorWithExposedPorts


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
PROJECT_DIR=os.environ["PROJECT_DIR"]


params = {
    'image_name'          : IMAGE,
    'dataset_name'        : 'iris.csv',
    'model_name'          : IMAGE + '.pkl',
    'param_test_size'     : 0.2,
    'param_random_state'  : 0,
    'param_max_iter'      : 100,
}

with DAG(
    'TrainingPipeline',
    default_args=default_args,
    params=params,
    catchup=False,
) as dag:

    t0 = DockerOperatorWithExposedPorts(
        task_id='dataset_step',
        image='ubuntu',
        auto_remove=True,
        command=[
            'cp',
            '/home/input/{{ params.dataset_name }}',
            '/home/output/{{ params.dataset_name }}'],
        mount_tmp_dir=True,
        mounts=[
            Mount(
                source=PROJECT_DIR +  '/bucket/data',
                target='/home/input/',
                type='bind'
            ),
            Mount(
                source=PROJECT_DIR + \
                    '/airflow/logs/TrainingPipeline/dataset_step/' + \
                    '{{ execution_date }}/',
                target='/home/output/',
                type='bind'
            )
        ]
    )


    t1 = DockerOperatorWithExposedPorts(
        task_id='training_step',
        image="{{ params.image_name }}",
        auto_remove=True,
        entrypoint=['sh','train.sh'],
        mount_tmp_dir=True,
        mounts=[
            Mount(
                source = PROJECT_DIR + \
                    '/airflow/logs/TrainingPipeline/dataset_step/' + \
                    '{{ execution_date }}/',
                target='/home/input/',
                type='bind'
            ),
            Mount(
                source = PROJECT_DIR + \
                    '/airflow/logs/TrainingPipeline/training_step/' + \
                    '{{ execution_date }}/',
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

    t2 = DockerOperatorWithExposedPorts(
        task_id='register_model',
        image='ubuntu',
        auto_remove=True,
        command=[
            'cp',
            '/home/input/{{ params.model_name }}',
            '/home/output/{{ execution_date }}_{{ params.model_name }}'],
        mount_tmp_dir=True,
        mounts=[
            Mount(
                source = PROJECT_DIR + \
                    '/airflow/logs/TrainingPipeline/training_step/' + \
                    '{{ execution_date }}/',
                target='/home/input/',
                type='bind'
            ),
            Mount(
                source=PROJECT_DIR + '/bucket/model',
                target='/home/output/',
                type='bind'
            ),
        ]
    )

    t0 >> t1 >> t2
