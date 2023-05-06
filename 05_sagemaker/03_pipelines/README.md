# Sesión 2 - Introducción a Sagemaker Pipelines

## Introducción

A lo largo de esta sesión veremos qué es Sagemaker y Sagemaker Pipelines y como nos pueden ayudar para desarrollar y desplegar modelos de machine learning en AWS de manera sencila y escalable. Finalmente veremos un ejemplo práctico de canalización

#### Sagemaker Pipelines

Dentro del conjunto de servicios que componen Sagemaker, Sagemaker Pipelines es el servicio de integración continua y MLOps de los modelos en Sagemaker. De esta forma, Sagemaker Pipelines permite orquestar flujos de Machine Learning por madio de "canalizaciones", donde cada uno de los procesos de del flujo es un "paso" dentro de la canalización.


## Definición de la canalización

Una canalización es un grafo directo acíclico (DAG) donde se definen cada uno de los procesos requerido por el flujo de creación del modelo como nodos del grafo. Del mismo modo, por medio de las conexiones del grafo se pueden definir las dependencias y orden de ejecución entre los diferentes procesos del flujo. En nuestro caso, el flujo que vamos a programar va a ser el siguiente

```
  processing    # Preparación de datos para entrenar
      |
    train       # Entrenamiento del modelo
      |
 create model   # Creación del modelo en Sagemaker para que pueda ser desplegado
      |
   transform    # Ejecución del modelo sobre un dataset
      |
register model  # Registro del modelo para dar Gobernabilidad (métricas de entrenamiento y despliegue)
```


Este es un grafo bastante sencillo en el que solo hay un camino, no obstante Sagemaker Pipelines permite programar diferentes caminos, según qué resultado den los pasos anteriores (como por ejemplo las métricas de entrenamiento). Para ello es necesario usar los objetos **ConditionalSteps** del sdk de Sagemaker Pipelines.

Aunque no entraremos en detalle en cada uno de ellos, el conjunto de **Steps** que están disponibles en Sagemaker Pipelines es el siguiente:

* **ProcessingStep** : Encapsula un trabajo de procesamiento
* **TrainingStep** : Encapsula un trabajo de entrenamiento
* **TuningStep** : Encapsula una búsqueda de hiperparámetros
* **CreateModelStep** : Crea un modelo dado un artefacto de un entrenamiento
* **RegisterModelStep** : Registra un modelo en Sagemaker 
* **TransformStep** : Encapsula un trabajo de predicción en batch
* **ConditionStep** : Elige el siguiente paso según el resultado de una condición
* **CallbackStep** : Encapsula la llamada a un servicio
* **LambdaStep** : Inicia la ejecución de una función AWS Lambda
* **ClarifyCheckStep** : Llama al servicio de AWS Sagemaker Clarify
* **QualityCheckStep** : 
* **EMRStep**
* **FailStep**

#### Procesamiento

En nuestro caso, el procesamiento va a hacer simplemente un split de los datos de entrenamiendo en el grupo de `train` y `test`. Al igual que el ejemplo de la sesión anterior, este script se puede ejecutar de manera local. La única particularidad con respecto al ejemplo de la sesión anterior es que ahora, el path por defecto lo hemos definido en `/opt/ml/processing`. Cualquier archivo que se monte dentro del contenedor de Sagemaker, tiene que estar dentro de esta ruta. 

```python
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split


base_dir = "/opt/ml/processing"

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="Species")
    parser.add_argument("--intput-file", type=str, default="iris.csv")
    parser.add_argument("--input-dir", type=str, default=base_dir+"/input")
    parser.add_argument("--output-dir", type=str, default=base_dir+"/output")

    # Read data
    df = pd.read_csv(os.path.join(args.train_dir, args.train_file))
    X = df.loc[:, df.columns != args.target]
    y = df.loc[:, df.columns == args.target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=28)

    # Save data
    pd.concat([X_train, y_train], axis=1).to_csv(
        os.path.join(args.output_dir, "train.csv"),
        index=False)

    pd.concat([X_test, y_test], axis=1).to_csv(
        os.path.join(args.output_dir, "test.csv"),
        index=False)
```

Dado un script de procesamiento, podemos ejecutarlo con Sagemaker usando un objeto **Processor**. Este objeto,

```
from sagemaker.sklearn.processing import SKLearnProcessor

sklearn_processor = SKLearnProcessor(
    framework_version="0.23-1",
    instance_type="ml.c4.xlarge",
    instance_count=1,
    role=role,
)
```

Para poder incluir este proceso como un paso dentro de la canalización, tenemos que definir el paso de procesamiento con el sdk de Sagemaker Pipelines usando un objeto de tipo **Step**. Concretamente, para un proceso de procesamiento: **ProcessingStep**.

```
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep


step_process = ProcessingStep(
    name="PreprocessData",
    processor=sklearn_processor,
    inputs=[
        ProcessingInput(
            source=input_data,
            destination="/opt/ml/processing/input"),
    ],
    outputs=[
        ProcessingOutput(
            output_name="train",
            source="/opt/ml/processing/train"),
        ProcessingOutput(
            output_name="test",
            source="/opt/ml/processing/test"),
    ],
    code="src/process.py",
)
```

Los argumentos que recive este objeto son:

* `processor`: Este argumento espera recivir un objeto de tipo **Processor**, en nuetros caso el **SKLearnProcess** que hemos definido anteriormente.

* `ProcessingInput` y `ProcessingOutput`: Sirven que Sagemaker cree los canales para el nodo del procesamiento de la canalización. Estos canales permiten montar archivos de S3 para leer los datos de entreda y guardar los datos ya procesados. Es importante que los paths dentro del contenedor de los canales estén dentro de la ruta `/opt/ml/processing`.

* `code`: Este campo solo es necesario en el caso de que no se haya especificado el `entry_point` en el **Processor**. Espera un path al archivo con el script de procesamiento definido anteriormente.


#### Entrenamiento

Para este entrenamiento usaremos un script de training similar al de la sesión anterior, solo que en este caso, en vez de separar los datos de entrenamiento e inferencia dentro del entrenamiento, estos vendrán ya separados en dos canales de input distintos. Además, en este caso no es necesario definir la función `model_fn(model_dir)` para cargar el modelo.

```python
import os
import json
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


base_dir = "/opt/ml"

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="Species")
    parser.add_argument("--n-estimators", type=int, default=10)
    parser.add_argument("--min-samples-leaf", type=int, default=3)

    parser.add_argument("--model-dir", type=str, default=base_dir+"/")
    parser.add_argument("--train-dir", type=str, default=base_dir+"/train")
    parser.add_argument("--test-dir", type=str, default=base_dir+"/test")

    args, _ = parser.parse_known_args()

    # Leemos los datos
    df_train = pd.read_csv(os.path.join(args.train_dir, 'train.csv'))
    X_train = df.loc[:, df.columns != args.target].values
    y_train = df.loc[:, df.columns == args.target].values.ravel()

    # Entrenamiento
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf
    ).fit(X_train, y_train)

    # Testeo
    df_tests = pd.read_csv(os.path.join(args.test_dir, 'test.csv'))
    X_test = df.loc[:, df.columns != args.target].values
    y_test = df.loc[:, df.columns == args.target].values.ravel()
    print("Accuracy=" + str(accuracy_score(y_test, model.predict(X_test))))

    # Guardamos el modelo
    with open(os.path.join(args.model_dir,'model.pkl'), "wb") as out:
        pickle.dump(model, out)
```

Creamos el objeto **Estimator** a partir del script de entrenamiento.

```python
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.steps import TrainingStep
from sagemaker.inputs import TrainingInput

sklearn_estimator = SKLearn(
    source_dir='./src',
    entry_point='train.py',
    framework_version='0.23-1',
    instance_type="ml.c4.xlarge",
    role=role,
    sagemaker_session=sesion,
    hyperparameters={
        "min_leaf_nodes": 3,
        "n_estimators": 10,
        "target": "Species"
    }
)
```

Y con esto creamos el **Step** de entrenamiento

```python
step_train = TrainingStep(
    name="TrainSklearnModel",
    estimator=sklearn_estimator,
    inputs={
        "train": TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            content_type="text/csv",
        ),
        "test": TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
            content_type="text/csv",
        ),
    },
)
```

En este caso, como vemos, los argumentos de `input` no son literales con paths a S3, sino que referencian a los resultados del **Step** de procesamiento. De esta forma, podemos definir el flujo de datos que pasa de un cada step del pipeline al siguiente durante el tiempo de ejecución del pipeline, a pesar de que durante la construcción del pipeline, ese archivo de s3 no exista.


#### Creación del modelo

Una vez el modelo ha sido entrenado, es necesario definir el **Step** de creación del modelo. Esto permite que el modelo esté almacenado en Sagemaker como un modelo entrenado para poder ser usado posteriormente para hacer inferencia

```python
from sagemaker.sklearn.model import SKLearnModel

model = SKLearnModel(
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    source_dir='./src',
    entry_point='train.py',
    framework_version='0.23-1',
    role=role,
    sagemaker_session=sesion,
)
```

Para crear el **Step** de creación del modelo, primero hay que crear un objeto **Model** de Sagemaker. Este objeto es similar a un **Estimator** la diferencia está en que en vez de servir para entrenar y desplegar un modelo, directamente carga un modelo ya entrenado en otra sesión para hacer inferencia.

```python
from sagemaker.inputs import CreateModelInput
from sagemaker.workflow.step_collections import CreateModelStep

step_create_model = CreateModelStep(
    name="CreateModel",
    model=model,
    inputs = CreateModelInput(instance_type="ml.m5.large")
)
```

El **Step** de creación de modelo, recibe argumento el **Model** definido con el artefacto que resultado del entrenamiento anterior. De esta forma, cualquier despliegue de este modelo está vinculado a ese entrenamiento en particular independientemente de los futuros entrenamientos que se hagan.

En el caso de que se quiera incluir como modelo no únicamente el modelo generado en el **Step** de entrenamiento sino también el de procesamiento prevido al entrenamiento, como una parte más del modelo, se puede hacer creando un objeto **PipelineModel** en vez de un **Model**. Aunque no lo veremos para este caso.

#### Inferencia

Una vez se tiene el modelo entrenado, ya vimos en la sesión anterior que Sagemaker permitía hacer la inferencia tanto online por medio de la creación de un Endpoint como en batch. Además de esto, Sagemaker Pipeline permite otros tipos de despligue, como por ejemplo por medio de funciones Lambda usando un **LambdaStep** o llamar a otros servicios de AWS con un **CallbackStep**. En esta sesión vamos a hacer una inferencia en batch.

```python
from sagemaker.transformer import Transformer


transformer = Transformer(
    model_name=step_create_model.properties.ModelName,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    output_path=output_data)
)
```

Creamos un objeto **Transform** de Sagemaker con el modelo creado en el **CreateModelStep** anteiror. El argumento `model_name` hace referencia al nombre del modelo creado en el paso anterior, pero podemos cogerlo dinámicamente con las propiedades del objeto. El argumento `output_path` hace referencia a la ruta del bucket de S3 para guardar los resultados.

```python
from sagemaker.inputs import TransformInput
from sagemaker.workflow.steps import TransformStep


step_transform = TransformStep(
    name="AbaloneTransform", 
    transformer=transformer,
    inputs=TransformInput(data=batch_data)
)
```

## Creación del Pipeline

Finalmente, una vez que tenemos todos los objetos **Step** que queremos ejecutar en nuestra canalización, solo nos queda crear el pipeline

```python
from sagemaker.workflow.pipeline import Pipeline


pipeline_name = f"AbalonePipeline"
pipeline = Pipeline(
    name=pipeline_name,
    steps=[
        step_process,
        step_train,
        step_create_model,
        step_transform
    ],
)

pipeline.upsert(role_arn=role)
definition = json.loads(pipeline.definition())
```

Para iniciar el procesamiento solo hay que llamar el método `start()`

```python
try:
    execution = pipeline.start()
    execution.wait()
    pprint(execution.list_steps())
except:
    pprint(execution.list_steps())
```

Esto inicial el procesamiento en Sagemaker. Cada uno de los **Steps** se puede ver desde la consola de Sagemaker en AWS en cada una de sus secciones (processing, training, inference,...). El pipeline se va ejecutando en secuencia hasta llegar el último **Step**.

## Conclusiones

Hemos visto que con Sagemaker Pipeline podemos orquestrar la secuencia de pasos que lleva el entrenamiento y despliegue de un modelo de Machine Learning. Aunque el ejemplo que hemos hecho es un pipeline sencillo, Sagemaker Pipelines te permite añadir nodos de control de flujo con los **ConditionalStep** con los que se puede programar y automatizar un CI/CD de machine learning.

Sin embargo, esta solución tiene algunas limitaciones a nivel de plataforma. Los objetos **Estimators** y **Processors**, aunque permiten usar diferentes frameworks de Machine Learning (sklearn, tensorflow, pytorch,..) e instalar dependencias, pueden dar error en algunas versiones de algunas dependencias. Además, no permite usar otros lenguajes de programación que no sea Python3 (R por ejemplo) ni tampoco usar tus propios framework de ML. En la siguiente sesión veremos cómo resolver esto último.

#### Referencias

* https://docs.aws.amazon.com/sagemaker/latest/dg/define-pipeline.html

* https://sagemaker.readthedocs.io/en/v2.23.4.post0/overview.html

* https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-pipelines/tabular/train-register-deploy-pipeline-model/train%20register%20and%20deploy%20a%20pipeline%20model.ipynb
