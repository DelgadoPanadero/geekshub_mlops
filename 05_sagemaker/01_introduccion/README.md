# Sesión 0 - Introducción a Sagemaker Notebook Instances

## Introducción

Antes de empezar a trabajar con Sagemaker es necesario instalar una serie de dependencias y crear el conjunto de recursos necesarios en AWS para poder ejecutar todo el contenido de las sesiones. En esta sesión se da por hecho que se tiene instalado Python3 y Docker.




## Creación de la cuenta

Antes que nada tenemos que tener una cuenta de AWS para poder acceder al conjunto de servicios de AWS. Esta cuenta deberá tener los permisos de la policy `AWSSagemakerFullAcess`. Esta policy, además de poder crear jobs de AWS Sagemaker, nos permitirá acceder a otros recursos que pueden ser necesarios para poder explotar todas las capacidades de Sagemaker, como por ejemplo:

* **AWS S3**: Para el guardado y lectura de artefactos durante los procesos.
* **AWS Glue**: Para ser usado como backend en los jobs de procesamiento de datos con Spark
* **AWS CloudWatch**: Para la monitorización de los procesos de entrenamiento.
* **AWS EC2**: Para el despliegue endpoints de inferencia de modelos



## Instalación de AWS cli y Sagemaker SDK


Una vez tenemos una cuenta creada con los roles necesarios, lo único que necesitamos tener instaladas y configuradas las herramientas de línea de comando y el SDK de Sagemaker en Python para la ejecución de procesos.

#### AWS Cli

Empezamos instalando la herramienta de AWS desde línea de comando (AWS Cli) siguiendo la documentación de [AWS](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html). En el caso de tener un sistema operativo Linux, la instalación se puede hacer con los siguientes comandos.

```
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

#### Sagemaker SDK

El código que se va a ejecutar a lo largo de las sesiones es Python3, además, es el único lenguaje soportado por Sagemaker para la ejecución de procesos de Machine Learning a más alto nivel. En los ejemplos de cada una de las sesiones, programaremos desde el entorno local la ejecución de procesos de Sagemaker en cloud, es por ello por lo que tenemos que tener el SDK de Sagemaker instalado

```
pip install sagemaker
```

Esto además, nos instala una serie de dependencias que también serán necesarias, como `boto3`


#### Dependencias de Python

Además de esto los ejemplos están construidos sobre notebooks de Jupyter y para poder ejecutarlos es necesario tener instalado Jupyter. La manera más sencilla de instalar Jupyter es por medio del siguiente comando:

```
pip install jupyter
```

Con Jupyter ya instalado, para poder visualizar y ejecutar los ejemplos de las sesiones, basta con levantar un servidor de jupyter usando el siguiente comando

```
jupyter notebook
```

Desde su interfaz podremos ver todas las sesiones así como cargar los notebooks y ejecutarlos.


## Creación y configuración de recursos 

#### Configuración de AWS CLi

Para que el AWS Cli apunte a nuestra cuenta es necesario configurarlo. Para ello basta con ejecutar el siguiente comando

```
aws configure
```

Esto nos pedirá el `ID Access Key` Y el `Acess Key Secret` de nuestra cuenta, además nos pedirá seleccionar la región en la cual se crearan todos los recursos que se generen desde el `AWS Cli`. En nuestro caso elegiremos la región más cercana a Madrid `eu-west-1`


#### Creación de un rol de ejecución

Los procesos de Sagemaker programados desde el SDK requieren el argumento `rol_arn`, que deberá ser un string con el ARN de un rol de AWS con los los permisos necesario de la policy `AWSSagemakerFullAcess`. Esto es necesario para que Sagemaker nos permita programar y ejecutar procesos desde el SDK.

Este rol se puede crear en el recurso de AWS IAM desde la consola de AWS

#### Creación AWS S3 y y AWS ECR

En los ejemplos de las siguientes sesiones usaremos los recursos de S3 para la lectura y estricuta de datos durante los procesos de ejecución de Sagemaker, así como la instanciación de imágenes Docker repositadas en


## Subida de Datos a S3

En las sesiones posteriores veremos como entrenar y desplegar modelos con Sagemaker. En estos ejemplos usaremos el dataset de iris y para que pueda ser usado por Sagemaker tenemos que subirlo a AWS S3. Esto se puede hacer con el siguiente código


```python
import boto3
import pandas as pd
import numpy as np
import sagemaker
from sklearn.datasets import load_iris


sesion = sagemaker.Session()
region = sesion.boto_session.region_name
bucket = sesion.default_bucket()

data = load_iris()
df = pd.DataFrame(
    data=np.c_[data['data'], data['target']],
    columns= data['feature_names'] + ['Species']
    )
    
df.to_csv("iris.csv",index=False)

s3_path = sesion.upload_data(
    path="iris.csv",
    bucket=bucket,
    key_prefix="curso_sagemaker/data"
)

print(s3_path)
```
