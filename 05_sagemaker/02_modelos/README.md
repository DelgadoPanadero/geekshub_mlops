# Sesión 1 - Introducción a Sagemaker (Sagemaker y Sagemaker Pipelines)

## Introducción

A lo largo de esta sesión veremos qué es Sagemaker y Sagemaker Pipelines y como nos pueden ayudar para desarrollar y desplegar modelos de machine learning en AWS de manera sencila y escalable. Finalmente veremos un ejemplo práctico de canalización

#### Sagemaker

Sagemaker es la plataforma de Machine Learning de AWS para su desarrollo y ejecución en la nube. Para ello proveé diferentes servicios y niveles de abstracción según las necesidades de cada proyecto. Desde un nivel más bajo de abstracción se puede usar simplemente para proveer instancias de cómputo con todos los requirimientos de software y hardware para entrenar y desplegar modelos de ML, hasta un nivel más alto en la que simplemente se despliegan modelos ya preentrenados de manera automática.

Aunque no entraremos en detalle en cada uno de ellos, el conjunto de objetos que están disponibles en Sagemaker desde su sdk para dar ese nivel de abstracción es el siguiente:

* **Estimators**: Encapsula el entrenamiento en Sagemaker
* **Models**: Encapsula modelo de ML ya entrenados
* **Predictors**: Provee de un sistema de inferencia en tiempo real desde Python accesiendo al servicio de Sagemaker Endpoint
* **Session**: Provee un conjunto de métodos y funcionalidades para trabajar con recursos de Sagemaker.
* **Transformers**: Encapsula la ejecución de trabajos de inferencia en batch en Sagemaker
* **Processors**: Encapsula la ejecución de trabajos de procesamiento de datos en Sagemaker

Una de las principales ventajas del uso de Sagemaker como herramienta de desarrollo de modelo de machine learning en cloud es su integración automática con otros recursos de almacenamiento de datos (como AWS S3, AWS DynamoDB, AWS Kinesis) como de motores de cómputo (Spark).



## Modelo de predicción con Iris

Vamos a ver como crear un modelo de clasificación que se ejecute en AWS para el dataset de Iris. Por el momento, vamos a ver como se podría entrenar y desplegar el modelo usando el SDK de Sagemaker para crear instancias de cómputo de Sagemaker (sin ver de momento como podemos orquestar y automatizar ese proceso). Para ello tenemos que crear un script de entrenamiento que en nuestro caso este script es `sesion_1/src/train.py`. 

```python
import os
import argparse

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="Species")
    parser.add_argument("--n-estimators", type=int, default=10)
    parser.add_argument("--min-samples-leaf", type=int, default=3)

    parser.add_argument("--train-file", type=str, default="iris.csv")
    parser.add_argument("--train-dir", type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--model-dir", type=str,
        default=os.environ.get("SM_MODEL_DIR"))

    args, _ = parser.parse_known_args()

    df = pd.read_csv(
        os.path.join(args.train_dir, args.train_file)
    )

    X = df.loc[:, df.columns != args.target].values
    y = df.loc[:, df.columns == args.target].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=28)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf
    ).fit(X_train, y_train)
    print("Accuracy=" + str(accuracy_score(y_test, model.predict(X_test))))

    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, path)
```

## Consideraciones del script de entrenamiento en Sagemaker

Este script de entrenamiento ese similar a cualquier otro script que se ejecutaría en un entorno local (lectura de datos, procesamiento y entrenamiento). No obstante, hay que tener en cuenta una serie de consideraciones para usar Sagemaker de manera correcta

#### Hiperparámetros

Al igual que como es común en un script de entrenamiento en local, los hiperparámetros se pueden pasar como argumentos de script.

```python
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="Species")
    parser.add_argument("--n-estimators", type=int, default=10)
    parser.add_argument("--min-samples-leaf", type=int, default=3)
```

#### Datos de entrada (Canales)

Posteriormente veremos que se puede montar un directorio de S3 dentro del contenedor que ejecuta el entrenamiento. Cada uno de los directorios montados se denomina "channel" y son accesibles desde la variable de entorno `SM_CHANNEL_[nombre del channel]`. Puede haber diferente número de canales, pero los mas común es existan estos dos:

* `SM_CHANNEL_TRAIN`: path del directorio con los datos de entrenamiento
* `SM_CHANNEL_TEST`: path del directorio con los datos de testeo.

```python
    parser.add_argument("--train-dir", type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--model-dir", type=str,
        default=os.environ.get("SM_MODEL_DIR"))

    args, _ = parser.parse_known_args()

    df = pd.read_csv(
        os.path.join(args.train_dir, args.train_file)
    )
```


#### Guardado del modelo

Para que el modelo guardado se pueda desplegar posteriormente con Sagemaker, el script debe guardar el modelo en un path concreto. Este path es accesible desde código por medio de la variable de entorno del sistema `SM_MODEL_DIR`.

```python
    parser.add_argument("--model-dir", type=str,
        default=os.environ.get("SM_MODEL_DIR"))

    ...

    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, path)
```

Además, se pueden guardar otros artefactos del modelo en el path dado por `SM_OUTPUT_DATA_DIR`.


#### Funciones del modelo

Para que Sagemaker posteriormente pueda cargar y ejecutar el modelo guardado en el entrenamiento para hacer inferencia, es necesario definir dentro del script qué funciones tiene que usar. Las funciones necesarias son:

* `def model_fn(model_dir)`
* `def input_fn(request_body, request_content_type)`
* `def predict_fn(input_object, model)`
* `def output_fn(prediction, content_type)`

De todas estas funciones la única que es obligatoria es `model_fn`, en al que se define cómo hay que cargar el modelo guardado. Todas las demás son opcionales.

```python
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf
```

## Ejecución con Sagemaker

#### Inicio de sesión con sagemaker

Para poder ejecutar crear jobs de AWS desde el SDK es necesario crear primero una sesión de Sagemaker. Esta sesión es simplemente un objeto que permite autenticarse en AWS desde código para poder acceder a los servicios de Sagemaker bajo el rol de la cuenta.

```python
import sagemaker

role = sagemaker.get_execution_role()

sesion = sagemaker.Session()
bucket = sesion.default_bucket() 
region = sesion.boto_session.region_name
```

Esta sesión, además también nos permite acceder a otros servicios vinculados a Sagemaker como puede ser un bucket de AWS S3 creado por defecto para este proceso.


## Entrenamiento con Sagemaker

#### Creación del Estimador
Los procesos en Sagemaker se pueden programar usando instancias del objetodo **Estimator** del SDK de Sagemaker. Existen diferentes tipos de Estimators para diferentes frameworks de machine learning (Sklearn, Tensorflow, Pytorch, Spark), así como diferentes modelos (KNN, PCA, RandomForest,...). Nuestro código crea un modelo de Scikit-Learn, por lo que usaremos el estimador SKLearn


```python
from sagemaker.sklearn.estimator import SKLearn

sklearn = SKLearn(
    source_dir='./src',
    entry_point='train.py',
    framework_version='0.23-1',
    instance_type="ml.c4.xlarge",
    role=role,
    sagemaker_session=sesion,
    metric_definitions=[
        {"Name": "train:accuracy", "Regex": "Accuracy=(.*)"}
    ],
    hyperparameters={
        "min_leaf_nodes": 3,
        "n_estimators": 10,
        "target": "Species" 
    }
)
```

* En el parámetro **entry_point** se especifica el path del archivo local que se quiere ejecutar. El objecto SKLearn se encarga de subirlo y ejecutarlo en una imágen Docker con scikit-learn instalado con la versión especificada.

* El parámetro **hyperparameters** recibe un diccionario donde se especifican los hiperparámetros para el entrenamiento. Estos hiperparámetros se pasará al script como argumentos de ejecución por lo que deberán estar definidos en el argparser del script anterior.

* En el parámetro **role** se debe pasar un role de AWS con permisos de ejecución de entrenamientos de Sagemaker. Para asegurarse que se tiene todos los acceso, se recomienda tener asignada la policy `AWSSagemakerFullAcess`. Del mismo modo, el parámetro **sesion** será necesario para desplegar posteriormente el modelo entrenado.

* El parámetro **metric_definition** nos permite definir métricas por medio de una expresión regular. Esto permite a Sagemaker saber, cuando se ejecuta un print, saber que se corresponde con el valor de una métrica de entrenamiento y recogerlo. Los valores de estas métricas se pueden visualizar en AWS CloudWatch.


#### Instalación de dependencias

Los contenedores que ejecutarn los scripts de entrenamiento en Sagemaker tienen acceso a algunas librerías preinstaladas según el tipo de **Estimator** que se haya usado (por ejemplo en nuestro caso, `scikit-learn`, `numpy` y `pandas`). Si es necesario instalar otras dependencias se pueden incluir añadiendo un un archivo `requirements.txt` dentro del directorio definido en el argumento`source_dir` cuando se instancia el objeto `SKLearn`.

```console
├── sesion_1/src
│    ├── requirements.txt
│    └── train.py
```

```python
sklearn = SKLearn(
    source_dir='./src',
    ...
    )
```

#### Fit method

Para iniciar la ejecución del script de entrenamiento en AWS hay que llamar el método `fit()` del objeto `SKLearn`. Este método espera recibir el argumento `inputs`, que espera recibir las rutas de S3 de los directorios donde se encuentran los archivos de entrenamiento.

```python
sklearn.fit(inputs={"train": PATH DE LA LOCALIZACIÓN DE S3})
```

En este diccionario se pueden definir múltiples elementos con diferentes rutas a directorios a S3. Cada uno de los elementos del diccionarios es un **canal** de entrada de datos. Sagemaker se encarga de montar el directorio de S3 definido, dentro del contenedor en una ruta que. Como comentábamos anteriormente la ruta accesible por el script por medio de la variable `SM_CHANNEL_[nombre del channel]`. En nuestro caso, siendo el nombre del canal `train`, podemos acceder a la ruta donde se han motado los datos por medio de

```python
os.environ.get("SM_CHANNEL_TRAIN")
```

El resultado de este proceso guarda el modelo generado en AWS S3 además listo para ser instanciado.


## Despliegue con Sagemaker

Una de las ventajas de Sagemaker es que, una vez generado el modelo, no es necesario crear un script para el despliegue del modelo. Dependiendo de tipo de **Estimator** que se ha ya usado para generar el modelo, Sagemaker es capaz de crear por si mismo el proceso de inferencia tanto si es online como en batch.

#### Cargar modelos ya entrenados

Se puede desplegar directamente un modelo entrenado en la misma sesión de Sagemaker, no obstante, lo más general será  ejecutar modelos entrenados en otras sesiones de Sagemaker. Se pueden cargar modelos entrenads en sesiones anteriores de la siguiente forma

```python
import boto3
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel


sm_boto3 = boto3.client("sagemaker")

last_job_name = sm_boto3.list_training_jobs(
    )['TrainingJobSummaries'][0]['TrainingJobName']

artifact_s3_path = sm_boto3.describe_training_job(
    TrainingJobName=last_job_name)["ModelArtifacts"]["S3ModelArtifacts"]

sklearn = SKLearnModel(
    model_data=artifact_s3_path,
    source_dir='./src',
    entry_point='train.py',
    framework_version='0.23-1',
    role=role,
    sagemaker_session=sesion,
)
```


#### Online Predict

Una vez el entrenamiento ha acabado, se puede llamándo al método `deploy()` sobre un **Estimator** que ya ha entrenado un modelo. Esto crea un endpoint donde se encuentra alojado el modelo entrenado

```python
predictor = sklearn.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge"
)
```

El resultado devuelve un objeto de tipo **Predictor** que se puede usar para hacer inferencia en al endpoint llamando al método `predict()` con un array de numpy o una lista (también se puede obtener el nombre del endopint llamando a la propiedad `name` del objeto **Predictor**)

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X,y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=28)

response = predictor.predict(X_test)
```

Por defecto, el resultado de haber ejecutado la predicción devuelve un array de numpy. Una vez que no es necesario seguir usando el servicio, se puede eliminar ejecutando

```python
predictor.delete_endpoint()
```

#### Batch Predict

Del mismo modo también se puede hacer predicciones en batch. Para ello hay que crear un objeto transformer en vez de predictor.

```python
transformer = sklearn.transformer(
    instance_count=1,
    instance_type="ml.m5.xlarge"
)
```

Con este objeto se puede hacer predicciones en batch con 

```python
transformer.transform(batch_input_s3, content_type="text/csv").wait()
```

Esto genera las predicciones y se guardan en AWS S3 para poder descargarlas.


## Conclusiones

En esta sesión hemos aprendido como entrenar un modelo de Machine Learning en Sagemaker usando un objeto **Estimator** del SDK. Dependiendo del framework que se use para crear el modelo (scikit-learn, tensorflow, pytorch,...), Sagemaker se encarga de provisionar la plataforma correcta para el script de entrenamiento y simplificar el despliegue por medio de su SDK.

Como hemos visto, el modelo entrenado debe de usar alguno de los frameworks de Machine Learning definidos en Sagemaker, no sería posible aplicar este proceso para proyectos que dependiesen de nuevas librerías. Además tampoco podemos automatizar el proceso de despliegue.


#### References

* https://sagemaker.readthedocs.io/en/v2.23.4.post0/overview.html

* https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/using_sklearn.html#preparing-the-scikit-learn-training-script

* https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-python-sdk/scikit_learn_randomforest/Sklearn_on_SageMaker_end2end.ipynb

