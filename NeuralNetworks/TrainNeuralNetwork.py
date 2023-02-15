from NeuralNetworks.NeuralNetwork import NeuralNetwork
import os
import json
from NeuralNetworks.Preprocessing import PreprocessingDataset

# Подготовка датасета
ProjectDir = os.getcwd()
if os.path.exists(os.path.join(ProjectDir,'Datasets/ArtyomDataset.json')):
    file = open('Datasets/ArtyomDataset.json','r',encoding='utf-8')
    Preprocessing = PreprocessingDataset()
    DataFile = json.load(file)
    dataset = DataFile['dataset']
    TrainInput,TrainTarget = Preprocessing.PreprocessingText(Dictionary = dataset,mode = 'train')
    file.close()
else:
    raise RuntimeError

if os.path.exists(os.path.join(ProjectDir,'Settings/Settings.json')):
    file = open(os.path.join(ProjectDir,'Settings/Settings.json'),'r',encoding='utf-8')
    DataFile = json.load(file)
    CATEGORIES = DataFile['CATEGORIES']
    CATEGORIES_TARGET = DataFile['CATEGORIES_TARGET']
    file.close()
else:
    raise RuntimeError

def Train():
    if __name__ == '__main__':
        # Вызов класса нейросети
        network = NeuralNetwork(CATEGORIES,CATEGORIES_TARGET)
        # Вызов функции тренировки нейросети
        network.train(TrainInput,TrainTarget)
        network.load()
        # Функция для вызова нейросети
        network.predict(Preprocessing.PreprocessingText(PredictArray = ['скажи время'],mode = 'predict'))
        
if __name__ == '__main__':
    Train()
    