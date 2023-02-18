from NeuralNetwork import NeuralNetwork
import os
import json
from Preprocessing import PreprocessingDataset
import matplotlib.pyplot as plt
import mplcyberpunk
from loguru import logger
import multiprocessing
import sys

# Подготовка датасета
ProjectDir = os.path.dirname(os.path.realpath(__file__))
if os.path.exists(os.path.join(ProjectDir,'Datasets/ArtyomDataset.json')):
    file = open(os.path.join(ProjectDir,'Datasets/ArtyomDataset.json'),'r',encoding='utf-8')
    Preprocessing = PreprocessingDataset()
    DataFile = json.load(file)
    dataset = DataFile['dataset']
    TrainInput,TrainTarget = Preprocessing.PreprocessingText(Dictionary = dataset,mode = 'train')
    file.close()
else:
    raise FileNotFoundError

if os.path.exists(os.path.join(ProjectDir,'Settings/Settings.json')):
    file = open(os.path.join(ProjectDir,'Settings/Settings.json'),'r',encoding='utf-8')
    DataFile = json.load(file)
    CATEGORIES = DataFile['CATEGORIES']
    CATEGORIES_TARGET = DataFile['CATEGORIES_TARGET']
    file.close()
else:
    raise FileNotFoundError

if __name__ == '__main__':
    network = NeuralNetwork(CATEGORIES,CATEGORIES_TARGET)
    if sys.argv[1] and sys.argv[1] == "train":
        # Вызов класса нейросети
        network.load()
        network.train(TrainInput,TrainTarget)
        network.save()
        network.predict(PreprocessingDataset(BaseCategoryPredict = True).PreprocessingText(PredictArray = ["скажи время"],mode = 'predict'))
    else:
        while True:
            command = input(">>>")
            if command == "exit":
                agree = input("Are you sure?(yes/no/y/n)")
                if len(agree.split()) == 0 or agree == "y" or agree == "yes":
                    break
                else:
                    continue
            elif command == "load":
                network.load()
            elif command == "save":
                network.save()
            elif command == "sande":
                network.save()
                # Вывод графика ошибки нейросети
                plt.plot(network.LossArray)
                # Сохранение картинки с графиком
                plt.savefig(network.PathLossGraph)
                # Вызов функции проверки нейросети с последующим выводом количества правильных ответов в виде процентов
                Accuracy = network.score(TrainInput,TrainTarget)
                logger.info("Neural network was trained on the dataset.")
                logger.info(f"Accuracy: {Accuracy}")
                logger.info(f"Loss graph was saved at the path: {network.PathLossGraph}")
                break
            elif command == "train":
                # Вызов функции тренировки нейросети
                if network.Training == False:
                    TrainProcess = multiprocessing.Process(target=network.train, args=(TrainInput,TrainTarget))
                    TrainProcess.start()
                else:
                    print("Neural Network is already training")
            elif command == "predict":
                Input = input("Question >>>")
                # Функция для вызова нейросети
                network.predict(PreprocessingDataset(BaseCategoryPredict = True).PreprocessingText(PredictArray = [Input],mode = 'predict'))