FROM python:3.10

# Install al the required dependencies
COPY ./requirements.txt /home
WORKDIR /home
RUN  pip install -r requirements.txt

# Copy the folder path into the container
COPY ./ /home
