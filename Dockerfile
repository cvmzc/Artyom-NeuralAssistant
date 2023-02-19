FROM python:3.9-slim-buster
WORKDIR /app
COPY .  .
RUN pip3 install -r ./NeuralNetworks/requirements.txt
RUN cd NeuralNetworks && python3 TrainNeuralNetwork.py train