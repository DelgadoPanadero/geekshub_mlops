FROM python:3.8

# Instalación de dependencias
COPY ./requirements.txt /home
WORKDIR /home
RUN  pip install -r requirements.txt

# Copiar el proyecto a dentro del contenedor
COPY ./ /home

# Crear los comandos
RUN echo "#!/bin/bash\n/usr/local/bin/python -u /home/process.py" > /usr/bin/process
RUN echo "#!/bin/bash\n/usr/local/bin/python -u /home/train.py" > /usr/bin/train
RUN echo "#!/bin/bash\n/usr/local/bin/python -u /home/serve.py" > /usr/bin/serve

RUN chmod +x /usr/bin/process
RUN chmod +x /usr/bin/train
RUN chmod +x /usr/bin/serve
