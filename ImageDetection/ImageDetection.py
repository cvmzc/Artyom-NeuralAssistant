import numpy as np
from skimage.util.shape import view_as_windows
from numpy.lib.stride_tricks import as_strided
import pickle
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from rich.progress import track
import mplcyberpunk
from loguru import logger
import platform
hostname = (platform.uname()[1]).lower()
if hostname.startswith("rpi"):
    from LED_RPI import LED_Green,LED_Red,LED_Yellow,Clean

# Параметры для нейросети
learning_rate = 0.0002
EPOCHS = 50000
BATCH_SIZE = 50
MinimalThreshold = 0.3

plt.style.use("cyberpunk")
np.random.seed(0)

ProjectDir = os.getcwd()
logger.add(os.path.join(ProjectDir,'Logs/ImageDetection.log'),format="{time} {level} {message}",level="INFO",rotation="200 MB",diagnose=True)

def unpickle(file):
    with open(file, 'rb') as fo:
        myDict = pickle.load(fo, encoding='latin1')
    return myDict

# Подготовка датасета
TrainData = unpickle('Datasets/train.dataset')
TestData = unpickle('Datasets/test.dataset')
TrainInput = TrainData["data"][:200]
TestInput = TestData["data"][:200]
# MetaData = unpickle('meta')
TrainTarget =  TrainData['fine_labels'][:200]
TestTarget = TestData['fine_labels'][:200]

class Conv:
    
    def __init__(self, num_filters):
        self.num_filters = num_filters
        
        #why divide by 9...Xavier initialization
        self.filters = np.random.randn(num_filters, 3, 3)/9
    
    def iterate_regions(self, image):
        #generates all possible 3*3 image regions using valid padding
        
        h,w = image.shape
        
        for i in range(h-2):
            for j in range(w-2):
                im_region = image[i:(i+3), j:(j+3)]
                yield im_region, i, j
                
    def forward(self, input):
        self.last_input = input
        print(input.shape)
        h,w = input.shape
        
        output = np.zeros((h-2, w-2, self.num_filters))
        
        for im_regions, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_regions * self.filters, axis=(1,2))
        return output
    
    def backprop(self, d_l_d_out, learn_rate):
        '''
        Performs a backward pass of the conv layer.
        - d_L_d_out is the loss gradient for this layer's outputs.
        - learn_rate is a float.
        '''
        d_l_d_filters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_l_d_filters[f] += d_l_d_out[i,j,f] * im_region

        #update filters
        self.filters -= learn_rate * d_l_d_filters

        return None

class MaxPool:
    def iterate_regions(self, image):
        h, w, _ = image.shape
        
        new_h = h // 2
        new_w = w // 2
        
        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i*2):(i*2+2), (j*2):(j*2+2)]
                yield im_region, i, j
                
    def forward(self, input):
        
        self.last_input = input
        
        h, w, num_filters = input.shape
        output = np.zeros((h//2, w//2, num_filters))
        
        for im_region, i, j in self.iterate_regions(input):
            output[i,j] = np.amax(im_region,axis=(0,1))
            
        return output
    
    def backprop(self, d_l_d_out):
        '''
        Performs a backward pass of the maxpool layer.
        Returns the loss gradient for this layer's inputs.
        - d_L_d_out is the loss gradient for this layer's outputs.
        '''
        d_l_d_input = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0,1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        #if the pixel was the max value, copy the gradient to it
                        if(im_region[i2,j2,f2] == amax[f2]):
                            d_l_d_input[i*2+i2, j*2+j2 ,f2] = d_l_d_out[i, j, f2]
                            break;
        return d_l_d_input

class Softmax:
    def __init__(self, input_len, nodes):
        # We divide by input_len to reduce the variance of our initial values
        self.weights = np.random.randn(input_len, nodes)/input_len
        self.biases = np.zeros(nodes)
    
    def forward(self, input):
        
        self.last_input_shape = input.shape
        
        input = input.flatten()
        self.last_input = input
        
        input_len, nodes = self.weights.shape
        
        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals
        
        exp = np.exp(totals)
        return(exp/np.sum(exp, axis=0)) 
    
    def backprop(self, d_l_d_out, learn_rate):
        """  
        Performs a backward pass of the softmax layer.
        Returns the loss gradient for this layers inputs.
        - d_L_d_out is the loss gradient for this layers outputs.
        """
        
        #We know only 1 element of d_l_d_out will be nonzero
        for i, gradient in enumerate(d_l_d_out):
            if(gradient == 0):
                continue
            
            #e^totals
            t_exp = np.exp(self.last_totals)
            
            #Sum of all e^totals
            S = np.sum(t_exp)
            
            #gradients of out[i] against totals
            d_out_d_t = -t_exp[i] * t_exp/ (S**2)
            d_out_d_t[i] = t_exp[i] * (S-t_exp[i]) /(S**2)
            
            # Gradients of totals against weights/biases/input
            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weights
            
            #Gradients of loss against totals
            d_l_d_t = gradient * d_out_d_t
            
            #Gradients of loss against weights/biases/input
            d_l_d_w = d_t_d_w[np.newaxis].T @ d_l_d_t[np.newaxis]
            d_l_d_b = d_l_d_t * d_t_d_b  
            d_l_d_inputs = d_t_d_inputs @ d_l_d_t
            
            #update weights/biases
            self.weights -= learn_rate * d_l_d_w
            self.biases -= learn_rate * d_l_d_b
            return d_l_d_inputs.reshape(self.last_input_shape)

class DetectionNeuralNetwork:
    def __init__(self,CATEGORIES = None,CATEGORIES_TARGET = None) -> None:
        self.CATEGORIES = CATEGORIES
        self.CATEGORIES_TARGET = CATEGORIES_TARGET
        self.INPUT_DIM = 1024
        self.HIDDEN_DIM = 512
        self.OUTPUT_DIM = len(self.CATEGORIES_TARGET)
        self.GenerateWeights
        self.LossArray = []
        self.Loss = 0
        self.LocalLoss = 0.5
        self.PathLossGraph = os.path.join(ProjectDir,'Plots','ImageDetection_Loss.png')
        self.Accuracy = 0

    def GenerateWeights(self):
        if hostname.startswith("rpi"):
            LED_Red()
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

    def batch(self,iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    def OneHotEncoder(self,y):
        OneHotValue = np.zeros((y.size,y.max() + 1))
        OneHotValue[np.arange(y.size),y] = 1
        OneHotValue = OneHotValue.T
        return OneHotValue

    def FeedForward(self,Input):
        try:
            self.h1 = Input @ self.w1 + self.b1
            self.HiddenLayer = self.sigmoid(self.h1)
            self.o = self.HiddenLayer @ self.w2 + self.b2
            self.OutputLayer = self.softmax(self.o)
            return self.OutputLayer
        except Exception as Error:
            logger.error(f"Exception error: {Error}.")

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

    def train(self,Input,Target):
        if hostname.startswith("rpi"):
            LED_Yellow()
        logger.info("Neural network was started of training.")
        # Генераци весов нейросети по длине входного массива(датасета)
        self.INPUT_DIM = len(TrainInput[0])
        # print(self.INPUT_DIM)
        self.GenerateWeights()

        conv = Conv(8)
        pool = MaxPool()
        softmax = Softmax(13 * 13 * 8, 10)
        # Прохождение по датасету циклом for
        for epoch in track(range(EPOCHS), description='[green]Training model'):
            # for Input,Target in zip(TrainInput,TrainTarget):
                print(Input)
                # Input = np.array(Input).reshape(-1,1)
                # Target = np.array(Target).reshape(-1,1)
                print(Target)
                # for TrainInput,TrainTarget in zip(self.batch(TrainInput,BATCH_SIZE),self.batch(TrainTarget,BATCH_SIZE)):
                # Вызов функции для расчёта прямого распространения нейросети
                Input = conv.forward((Input/255) - 0.5)
                Input = pool.forward(Input)
                Input = softmax.forward(Input)
                PredictedValue = self.FeedForward(Input)
                # Вызов функции для расчёта обратного распространения ошибки нейросети
                gradient = np.zeros(10)
                gradient[Target] = -1/Input[Target]
                
                
                #Backprop
                gradient = softmax.backprop(gradient, learning_rate)
                gradient = pool.backprop(gradient)
                gradient = conv.backprop(gradient, learning_rate)
                self.BackwardPropagation(Input,Target)
                # Расчёт ошибки
                self.Loss = np.sum(self.CrossEntropy(self.OutputLayer, Target))
                # Сохранение модели с наименьшей ошибкой
                if epoch % (EPOCHS / 20) == 0 and self.Loss <= self.LocalLoss:
                    self.LocalLoss = self.Loss
                    self.save()
                # Добавление ошибки в массив для дальнейшего вывода графика ошибки нейросети
                self.LossArray.append(self.Loss)
        # Вывод графика ошибки нейросети
        plt.plot(self.LossArray)
        # plt.show() 
        # Сохранение картинки с графиком
        plt.savefig(self.PathLossGraph)
        # Вызов функции проверки нейросети с последующим выводом количества правильных ответов в виде процентов
        # self.Accuracy = self.score(TrainInput,TrainTarget)
        logger.info("Neural network was trained on the dataset.")
        # logger.info(f"Accuracy: {self.Accuracy}")
        logger.info(f"Loss graph was saved at the path: {self.PathLossGraph}")
        if hostname.startswith("rpi"):
            LED_Green()
    
    # Функция для вызова нейросети
    def predict(self,Input):
        PredictedArray = self.FeedForward(Input)
        PredictedValue = np.argmax(PredictedArray)
        if float(PredictedArray[0][PredictedValue]) >= MinimalThreshold:
            print(self.CATEGORIES_TARGET[str(PredictedValue)])
            return self.CATEGORIES_TARGET[str(PredictedValue)],PredictedValue
        else:
            print("don't know")
            # Если нейросеть не уверенна в своём ответе,то отправляется ответ в виде фразы 
            return "don't_know"
    
    # Функция проверки нейросети с последующим выводом количества правильных ответов в виде процентов
    def score(self,TestInput,TestTarget):
        correct = 0
        for Input,Target in zip(TestInput,TestTarget):
            PredictedValue = self.FeedForward(Input)
            Output = np.argmax(PredictedValue)
            if Output == Target:
                correct += 1
        accuracy = correct / len(TestInput)
        return accuracy
    
    # Сохранение весов и смещений нейросети
    def save(self,PathParametrs = os.path.join(ProjectDir,'Models','ImageDetection.npz')):
        np.savez_compressed(PathParametrs, self.w1,self.w2,self.b1,self.b2,EPOCHS,learning_rate,BATCH_SIZE,MinimalThreshold)
    
    # Загрузка весов и смещений нейросети
    def load(self,PathParametrs = os.path.join(ProjectDir,'Models','ImageDetection.npz')):
        ParametrsFile = np.load(PathParametrs)
        self.w1 = ParametrsFile['arr_0']
        self.w2 = ParametrsFile['arr_1']
        self.b1 = ParametrsFile['arr_2']
        self.b2 = ParametrsFile['arr_3']
        logger.info("Weights of neural network was loaded.")

if __name__ == "__main__":
    detection = DetectionNeuralNetwork(CATEGORIES_TARGET = TrainTarget)
    detection.train(TrainInput,TrainTarget)