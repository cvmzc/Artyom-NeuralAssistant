# Импортирование библиотек для обучения и проверки нейросети
import numpy as np
import os
import json
from Preprocessing import PreprocessingDataset
import matplotlib.pyplot as plt
from rich.progress import track
import mplcyberpunk
from loguru import logger

plt.style.use("cyberpunk")
np.random.seed(0)

ProjectDir = os.getcwd()
logger.add(os.path.join(ProjectDir,'Logs/Transformer_NeuralNetwork.log'),format="{time} {level} {message}",level="INFO",rotation="200 MB",diagnose=True)

# Подготовка датасета
if os.path.exists(os.path.join(ProjectDir,'Datasets/TransformerDataset.json')):
    file = open('Datasets/TransformerDataset.json','r',encoding='utf-8')
    Preprocessing = PreprocessingDataset()
    DataFile = json.load(file)
    dataset = DataFile['dataset']
    TrainInput,TrainTarget = Preprocessing.PreprocessingTransformer(Dictionary = dataset,mode = 'train')
    file.close()
else:
    raise RuntimeError

if os.path.exists(os.path.join(ProjectDir,'NeuralNetworkSettings/TransformerSettings.json')):
    file = open(os.path.join(ProjectDir,'NeuralNetworkSettings/TransformerSettings.json'),'r',encoding='utf-8')
    DataFile = json.load(file)
    # CATEGORIES = DataFile['CATEGORIES']
    CATEGORIES_TARGET = DataFile['CATEGORIES_TARGET']
    file.close()
else:
    raise RuntimeError


learning_rate = 0.001
EPOCHS = 50000
BATCH_SIZE = 50
MinimumThreshold = 0.6

class Transformer_NeuralNetwork:
    # Функция инициализации переменных
    def __init__(self,CATEGORIES:dict = {},CATEGORIES_TARGET:list = []):
        self.CATEGORIES = CATEGORIES
        self.CATEGORIES_TARGET = CATEGORIES_TARGET
        self.INPUT_DIM = 60
        self.HIDDEN_DIM = 512
        self.OUTPUT_DIM = len(CATEGORIES_TARGET)
        self.GenerateWeights()
        self.LossArray = []
        self.Loss = 0
        self.LocalLoss = 0.5
        self.PathLossGraph = os.path.join(ProjectDir,'Plots','Transformer_NeuralNetwork_Loss.png')
        self.Accuracy = 0

    # Функция для генерации весов нейросети
    def GenerateWeights(self):
        try:
            self.w1 = np.random.rand(self.INPUT_DIM, self.HIDDEN_DIM)
            self.b1 = np.random.rand(1, self.HIDDEN_DIM)
            self.w2 = np.random.rand(self.HIDDEN_DIM, self.OUTPUT_DIM)
            self.b2 = np.random.rand(1, self.OUTPUT_DIM)
            self.w1 = (self.w1 - 0.5) * 2 * np.sqrt(1/self.INPUT_DIM)
            self.b1 = (self.b1 - 0.5) * 2 * np.sqrt(1/self.INPUT_DIM)
            self.w2 = (self.w2 - 0.5) * 2 * np.sqrt(1/self.HIDDEN_DIM)
            self.b2 = (self.b2 - 0.5) * 2 * np.sqrt(1/self.HIDDEN_DIM)
        except Exception as Error:
            logger.error(f"Exception error: {Error}.")
    # Функция активации
    def relu(self,t):
        return np.maximum(t, 0)

    def softmax(self,t):
        out = np.exp(t)
        return out / np.sum(out, axis=1, keepdims=True)

    # Функция для расчёта ошибки
    def CrossEntropy(self,z, y):
        return -np.log(np.array([z[j, y[j]] for j in range(len(y))]))

    def to_full(self,y, num_classes):
        y_full = np.zeros((len(y), num_classes))
        for j, yj in enumerate(y):
            y_full[j, yj] = 1
        return y_full

    # Производная функции активации
    def deriv_relu(self,t):
        try:
            return (t >= 0).astype(float)
        except Exception as Error:
            logger.error(f"Exception error: {Error}.")
        
    # Функция активации
    def sigmoid(self,x):
        try:
            return 1 / (1 + np.exp(-x))
        except Exception as Error:
            logger.error(f"Exception error: {Error}.")
    # Производная функции активации
    def deriv_sigmoid(self,y):
        try:
            return y * (1 - y)
        except Exception as Error:
            logger.error(f"Exception error: {Error}.")
    # Функция прямого распространени нейросети
    def FeedForward(self,Input):
        try:
            self.h1 = Input @ self.w1 + self.b1
            self.HiddenLayer = self.sigmoid(self.h1)
            self.o = self.HiddenLayer @ self.w2 + self.b2
            self.OutputLayer = self.softmax(self.o)
            return self.OutputLayer
        except Exception as Error:
            logger.error(f"Exception error: {Error}.")
    # Функция обратного распространения ошибки нейросети
    def BackwardPropagation(self,Input,Target):
        try:
            y_full = self.to_full(Target, self.OUTPUT_DIM)
            d_o = self.OutputLayer - y_full
            d_w2 = self.HiddenLayer.T @ d_o
            d_b2 = np.sum(d_o, axis=0, keepdims=True)
            d_hl = d_o @ self.w2.T
            d_h1 = d_hl * self.deriv_sigmoid(self.h1)
            d_w1 = Input.T @ d_h1
            d_b1 = np.sum(d_h1, axis=0, keepdims=True)

            # Обновление весов и смещений нейросети
            self.w1 = self.w1 - learning_rate * d_w1
            self.b1 = self.b1 - learning_rate * d_b1
            self.w2 = self.w2 - learning_rate * d_w2
            self.b2 = self.b2 - learning_rate * d_b2
        except Exception as Error:
            logger.error(f"Exception error: {Error}.")

    def train(self,TrainInput,TrainTarget):
        logger.info("Neural network was started of training.")
        # Генераци весов нейросети по длине входного массива(датасета)
        # self.INPUT_DIM = len(TrainInput[0])
        print(self.INPUT_DIM)
        self.GenerateWeights()
        # Прохождение по датасету циклом for
        for epoch in track(range(EPOCHS), description='[green]Training Transformer Model'):
            # Вызов функции для расчёта прямого распространения нейросети
            PredictedValue = self.FeedForward(TrainInput)
            # Вызов функции для расчёта обратного распространения ошибки нейросети
            self.BackwardPropagation(TrainInput,TrainTarget)
            # Расчёт ошибки
            self.Loss = np.sum(self.CrossEntropy(self.OutputLayer, TrainTarget))
            # Сохранение модели с наименьшей ошибкой
            if self.Loss <= self.LocalLoss:
                self.LocalLoss = self.Loss
                self.save()
            # Добавление ошибки в массив для дальнейшего вывода графика ошибки нейросети
            self.LossArray.append(self.Loss)
        # Вывод графика ошибки нейросети
        plt.plot(self.LossArray)
        # plt.show() 
        # Сохранение картинки с графиком
        plt.savefig(self.PathLossGraph)
        print(f"Loss: {self.Loss}")
        # Вызов функции проверки нейросети с последующим выводом количества правильных ответов в виде процентов
        self.Accuracy = self.score(TrainInput,TrainTarget)
        logger.info("Neural network was trained on the dataset.")
        logger.info(f"Accuracy: {self.Accuracy}")
        logger.info(f"Loss graph was saved at the path: {self.PathLossGraph}")

    # Функция для вызова нейросети
    def predict(self,Input):
            print(Input)
            PredictedArray = self.FeedForward(Input[0])
            print(PredictedArray)
            PredictedValue = np.argmax(self.FeedForward(PredictedArray))
            print(PredictedValue)
            print(self.CATEGORIES_TARGET[str(PredictedValue)])
            # if float(PredictedArray[PredictedValue]) >= MinimumThreshold:
            #     print(self.CATEGORIES_TARGET[str(PredictedValue)])
            #     return self.CATEGORIES_TARGET[str(PredictedValue)]
            # else:
            #     # Если нейросеть не уверенна в своём ответе,то отправляется ответ в виде фразы 
            #     return "don't_know"
    
    # Функция проверки нейросети с последующим выводом количества правильных ответов в виде процентов
    def score(self,TrainInput,TrainTarget):
        correct = 0
        for Input,Target in zip(TrainInput,TrainTarget):
            PredictedValue = self.FeedForward(Input)
            Output = np.argmax(PredictedValue)
            if Output == Target:
                correct += 1
        accuracy = correct / len(TrainInput)
        # print(accuracy)
        return accuracy
    
    # Сохранение весов и смещений нейросети
    def save(self,PathParametrs = os.path.join(ProjectDir,'Models','Transformer_NeuralAssistant.npz')):
        np.savez_compressed(PathParametrs, self.w1,self.w2,self.b1,self.b2)
    
    # Загрузка весов и смещений нейросети
    def load(self,PathParametrs = os.path.join(ProjectDir,'Models','Transformer_NeuralAssistant.npz')):
        ParametrsFile = np.load(PathParametrs)
        self.w1 = ParametrsFile['arr_0']
        self.w2 = ParametrsFile['arr_1']
        self.b1 = ParametrsFile['arr_2']
        self.b2 = ParametrsFile['arr_3']
        logger.info("Weights of neural network was loaded.")

if __name__ == "__main__":
    network = Transformer_NeuralNetwork(CATEGORIES_TARGET=CATEGORIES_TARGET)
    network.train(TrainInput,TrainTarget)
    network.predict(Preprocessing.PreprocessingTransformer(PredictArray = ["поставь будильник на одиннадцать тридцать"],mode = 'predict'))