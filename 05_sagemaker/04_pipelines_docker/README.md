# Sesion 3 - Construye tus propios algoritmos

## Introduccion

#### Limitaciones de Sagemaker

Como hemos comentado en la sección anterior Sagemaker, así como Sagemaker Pipelines, te permiten ejecutar y definir flujos de CI/CD, de tus scripts de entrenamiento de modelos, de manera sencilla abstrayendo al desarrollador de la gestión de la plataforma. No obstante, esta abstracción está solo limitada a un conjunto de frameworks y versiones, e incluso aunque intentes instalar nuevas dependencias, las versiones de los paquetes están limitadas a las versiones que están accesibles en el instalador del contenedor.

Además de esto, la abstracción de la plataforma y el framework de desarrollo, hace que dos modelos que tengan exactamente el mismo flujo lógico de procesamiento, tengan que ser desplegados y ejecutados con dos códigos de Sagemaker Pipelines debido a que los objetos **Estimators** y **Processors** son diferentes.

#### Build your own image

La solución de Sagemaker para este conjunto de limitaciones es la de construir tu propia imagen Docker con todas las demendencias, de manera que en vez de desarrollar códigos para ejecutar en Sagemaker, se desarrollan aplicaciones contenidas en imágenes.

#### ¿Cuándo debería construir mi propio algoritmo?

1 - Cuando quieres usar otros lenguajes de programación diferentes a Python3 u otros frameworks distintos a los que te ofrece Sagemaker.

2 - Cuando quieres abstraer el script de creación y ejecución del pipeline del framework del modelo.

3 - Cuando el código del proyecto de machine learning es demasiado complejo como para que pueda estar todo contenido en una única carpeta.


## Construcción de algoritmo contenerizado

Para poder construir el algoritmo de manera correcta y que se ejecute en Sagemaker, hay que tener en cuenta algunas consideraciones

#### Esctructura de proyecto

Según el caso de uso, el proyecto puede tener diferentes estructuras de directorios, no obstante, de manera general, seguramente tenga una estructura parecida a la siguiente

```
project/
  ├── process/
  │     ├── utils/
  │     └── main.py
  ├── train/
  │     ├── utils/
  │     └── main.py
  ├── serve/
  │     ├── utils/
  │     └── main.py
  ├── requirements.txt
  ├── build_and_push.sh
  └── Dockerfile
```

#### Creación de la imagen

Para crear la imagen docker tenemos que programar un Dockerfile. El objetivo es conseguir que todo el proyecto quede autocontenido dentro de la imagen y para ello, este dockerfile deberá hacer al menos 3 cosas:

* Instalar las dependencias necesarias
* Copiar el proyecto dentro de la imagen
* Crear los comandos de `train` y `serve`.

Los comandos de `train` y `serve` son necesarios ya que son los que usan los objetos **Estimator** y **Transformer** para ejecutar el entrenamiento y la inferencia respectivamente cuando reciben una imagen Docker en vez de un script para ejectura en Sagemaker. En nuestro caso, por conveniencia también vamos a definir el comando de `processing` para llamarlo desde el el **Processor**, sin embargo, este último no es necesario. El dockerfile que vamos a usar para el ejemplo es el siguiente:
 

```
FROM python:3.8

# Instalación de dependencias
COPY ./requirements.txt /home
WORKDIR /home
RUN  pip install -r requirements.txt

# Copiar el proyecto a dentro del contenedor
COPY ./ /home

# Crear los comandos
RUN echo "#!/bin/bash\n/usr/local/bin/python -u /home/processing/main.py" > /usr/bin/processing
RUN echo "#!/bin/bash\n/usr/local/bin/python -u /home/train/main.py" > /usr/bin/train
RUN echo "#!/bin/bash\n/usr/local/bin/python -u /home/serve/main.py" > /usr/bin/serve

RUN chmod +x /usr/bin/processing
RUN chmod +x /usr/bin/train
RUN chmod +x /usr/bin/serve
```

Para que Sagemaker pueda utilizar esta imagen posteriormente tiene que estar disponible en AWS ECR. Para ello, tenemos que construir la imagen y subirla a ECR. En nuestro caso esto lo vamos a hacer con el script `build_and_push.sh`

```
#!/usr/bin/sh

# The name of our algorithm
algorithm_name=sagemaker-decision-trees
account=$(aws sts get-caller-identity --query Account --output text)
region=$(aws configure get region)
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"

# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null
fi

# Get the login command from ECR and execute it directly
aws ecr get-login-password --region ${region}|docker login --username AWS --password-stdin ${fullname}

# Build the docker image with the image name and then push it to ECR
docker build  -t ${algorithm_name} .
docker tag ${algorithm_name} ${fullname}
docker push ${fullname}
```

Con esto ya tendríamos creada la imagen Docker. Es importante destacar que el hecho de haber definido la ejecución de cada unos de los procesos con los comandos de `processing`, `train` y `serve`, nos permite conseguir que la ejecución de la imagen sea independiente del código que contiene (framework, lenguaje, dependencias,...). Por ejemplo, definiendo el comando `train` como comando de entrenamiento, nos da igual que internamente ejecute `python train/main.py` o `Rscript train/main.R`, por lo que la ejecución sería agnóstica al lenguaje de dentro.


#### Api de inferencia

En las sesiones anteriores, habíamos delegado en Sagemaker la responsabilidad de crear un servicio web para hacer inferencia o de ejecutar el modelo en batch. Esto era posible ya que estábamos usando las imágenes de Sagemaker que, dada una declaración de `def model_fn(model_dir)`, ejecutarla en una API rest de flask predefinida en la imagen. Si queremos crear nuestra propia imagen, tenemos que desarrollar la API para servir el modelo desde el contenedor. Los métodos que tiene que tener esta API son:

* **[GET] /ping**: este endpoint sirve a Sagemaker para comprobar que la imagen está levantada, simplemente debe devolver un 200.

* **[POST] /invocations**: este endpoint recibirá las peticiones de inferencia con los datos a predecir en el body de la petición. en el caso de que en el **Transformer** se haya definido un `ContentType` o un `Accept` lo recibirá también.

```python
import csv
import pickle
from io import StringIO

import flask
import pandas as pd


# cargamos modelo
model = joblib.load('/opt/ml/model/model.joblib')

app = flask.Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    return 200


@app.route("/invocations", methods=["POST"])
def predict():

    features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

    # Convert from CSV to pandas
    if flask.request.content_type == "text/csv":
        data = flask.request.data.decode("utf-8")
        s = StringIO(data)
        X = pd.read_csv(s)[features]

        # Do the prediction
        predictions = pd.DataFrame(
            model.predict_proba(X), columns=model.classes_)

        # Convert from numpy back to CSV
        out = StringIO()
        predictions.to_csv(out, header=False, index=False)
        result = out.getvalue()

        return flask.Response(
            response=result,
            status=200,
            mimetype="text/csv")


app.run(host="0.0.0.0", port=8080, debug=True)
```

Este código, además de levantar el servicio web y responder a las peticiones, deberá ser capaz de leer el modelo creado por el proceso de entrenamiento. Sagemaker se encarga de montar el artefacto generado por el objeto **Estimator** el directorio `/opt/ml/model` en el contenedor ejecutado por el objeto **Transformer**.


#### Procesamiento y Entrenamiento

Para el procesamiento y el entrenamiento del modelo usaremos los mismos scripts que la sesión anterior salvo por las variables de entorno. Como vimos en la sesion 1, las imágenes de Sagemaker contienen variables de entorno con configuración relativa a la plataforma como por ejemplo, los paths de lectura y escritura donde Sagemaker monta los buckets de AWS S3. En el caso de construir nosotros la imagen estas variables no existen por lo que debemos crearlas o definir los paths de lectura y escritura como literales dentro del modelo:

Para el caso de un trabajo de entrenamiento los paths usados por Sagemaker para montar archivos dentro del contenedor son:

```
/opt/ml
|-- input
|   |-- config
|   |   |-- hyperparameters.json
|   |   `-- resourceConfig.json
|   `-- data
|       `-- <channel_name>
|           `-- <input data>
|-- model
|   `-- <model files>
`-- output
    `-- failure
```


y para un objeto trabajo de procesamiento el path es cualquier direcotrio definido dentro de `/opt/ml/processing`



## Ejecución del algoritmo


#### Ejecución en local

#### Ejecución en Sagemaker

Del mismo modo que como vimos en sesiones anteriores, para ejecutar el proceso de entrenamiento hay que usar un objeto **Estimator**. En este caso, no será dependiente de ningún framework y además, en vez de especificar el código para que se suba y se ejecute en Sagemaker, especificaremos la url de la imagen que hemos subido a ECR. El objeto **Estimator** se encargará automáticamente de instanciar el contenedor llamando al comando **train**

```python
from sagemaker.estimator import Estimator


account = sesion.boto_session.client("sts").get_caller_identity()["Account"]
image = "{}.dkr.ecr.{}.amazonaws.com/sagemaker-decision-trees:latest".format(account, region)
tree = Estimator(
    image_uri=image,
    role=role,
    instance_count=1,
    instance_type="ml.c4.2xlarge",
    output_path="s3://{}/output".format(bucket),
    sagemaker_session=sesion,
)

tree.fit(inputs={
    "train": s3_train_path,
    "test": s3_test_path})
```

Del mismo modo que en el caso anterior, habiendo definido el script `serve.py` podemos ejecutar la inferencia por medio del método `deploy()` o `transform()`. En este caso usaremos el método `transform()` para hacer una predicción batch.

```python
from sagemaker.transformer import Transformer

output_path="s3://sagemaker-eu-west-1-827345860551/curso_sagemaker/output"

transformer = tree.transformer(
    instance_count=1,
    instance_type="ml.m4.xlarge",
    output_path=output_path,
    assemble_with="Line",
    accept="text/csv",
)

transformer.transform(
    s3_test_path,
    content_type="text/csv",
)
transformer.wait()
```


#### Ejecución en Sagemaker Pipelines

La ejecución con Sagemaker Pipelines será similar a la que vimos en la sesión anterior, la única diferencia está en que en cada uno de los steps especificaresmos el argumentos `image_uri` con la imagen que hemos creado en vez de el argumento `entry_point` con el path del código en local.


## Conclusiones

En esta sesión hemos visto cómo crear nuestros propios algoritmos de ML creando una imagen Docker autocontenida en vez de por medio de los objetos a alto nivel de Sagemaker. Esto nos permite tener todas las ventajas de Sagemaker y Sagemaker Pipelines (provisionamiento automático, CI/CD, monitorización,...) al mismo tiempo que nos da una mayor flexibilidad a la hora de desarrollar el modelo que luego se ejecutará en Sagemaker.

En nuestro caso, este modelo está hecho en Sklearn con Python, siguiendo el mismo modelo que los ejemplos abnteriores, no obstante gracias a las ventajas del uso de un modelo contenerizadp. nada nos inpediría usar cualquier otro framework o lenguaje de programación aunque no estuviese dentro de los lenguajes soportados por los objetos de abstracción de Sagemaker

#### Referencias

* https://sagemaker-examples.readthedocs.io/en/latest/advanced_functionality/scikit_bring_your_own/scikit_bring_your_own.html#Part-1%3A-Packaging-and-Uploading-your-Algorithm-for-use-with-Amazon-SageMaker
