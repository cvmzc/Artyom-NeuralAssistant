import os
import numpy as np
import matplotlib.pyplot as plt
import json
from Preprocessing import PreprocessingDataset
from rich.progress import track
import mplcyberpunk
from loguru import logger
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import platform
import pyaudio
import wave

hostname = (platform.uname()[1]).lower()
if hostname.startswith("rpi"):
    from LED_RPI import LED_Green,LED_Red,LED_Yellow,Clean

plt.style.use("cyberpunk")
np.random.seed(0)

ProjectDir = os.getcwd()
AudioDatasetDir = os.path.join(ProjectDir,"Datasets/SpeechDataset/")
logger.add(os.path.join(ProjectDir,'Logs/SpeechNeuralNetwork.log'),format="{time} {level} {message}",level="INFO",rotation="200 MB",diagnose=True)

learning_rate = 0.002
EPOCHS = 100000
BATCH_SIZE = 28

class SpeechRecognition:
    def __init__(self):
        self.INPUT_DIM = 20
        self.HIDDEN_DIM = 64
        self.OUTPUT_DIM = 256
        self.GenerateWeights()
        self.LossArray = []
        self.Loss = 0
        self.LocalLoss = 0.5
        self.PathLossGraph = os.path.join(ProjectDir,'Plots','Loss.png')
        self.Accuracy = 0
        self.LastOutputLayer = 0

    # Функция для генерации весов нейросети
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
        # try:
            return np.maximum(t, 0)
        # except Exception as Error:
        #     logger.error(f"Exception error: {Error}.")

    def softmax(self,t):
        # try:
            out = np.exp(t)
            return out / np.sum(out, axis=1, keepdims=True)
        # except Exception as Error:
        #     logger.error(f"Exception error: {Error}.")

    # Функция для расчёта ошибки
    def CrossEntropy(self,z, y):
        # try:
            return -np.log(np.array([z[j, y[j]] for j in range(len(y))]))
        # except Exception as Error:
        #     logger.error(f"Exception error: {Error}.")

    def to_full(self,y, num_classes):
        # try:
            y_full = np.zeros((len(y), num_classes))
            for j, yj in enumerate(y):
                y_full[j, yj] = 1
            return y_full
        # except Exception as Error:
        #     logger.error(f"Exception error: {Error}.")

    # Производная функции активации
    def deriv_relu(self,t):
        # try:
            return (t >= 0).astype(float)
        # except Exception as Error:
        #     logger.error(f"Exception error: {Error}.")
        
    # Функция активации
    def sigmoid(self,x):
        # try:
            return 1 / (1 + np.exp(-x))
        # except Exception as Error:
        #     logger.error(f"Exception error: {Error}.")
    # Производная функции активации
    def deriv_sigmoid(self,y):
        # try:
            return y * (1 - y)
        # except Exception as Error:
        #     logger.error(f"Exception error: {Error}.")
    
    def batch(self,iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    # Функция прямого распространени нейросети
    def FeedForward(self,Input):
        # try:
            self.h1 = Input @ self.w1 + self.b1
            self.HiddenLayer = self.relu(self.h1)
            self.o = self.HiddenLayer @ self.w2 + self.b2
            # self.OutputLayer = self.softmax(self.o)
            self.OutputLayer = self.o
            return self.OutputLayer
        # except Exception as Error:
        #     logger.error(f"Exception error: {Error}.")

    # Функция обратного распространения ошибки нейросети
    def BackwardPropagation(self,Input,Target):
        # try:
            y_full = self.to_full(Target, self.OUTPUT_DIM)
            # print(type(y_full))
            d_o = self.OutputLayer - y_full
            d_w2 = self.HiddenLayer.T @ d_o
            d_b2 = np.sum(d_o, axis=0, keepdims=True)
            d_hl = d_o @ self.w2.T
            d_h1 = d_hl * self.deriv_relu(self.h1)
            d_w1 = Input.T @ d_h1
            d_b1 = np.sum(d_h1, axis=0, keepdims=True)

            # Обновление весов и смещений нейросети
            self.w1 = self.w1 - learning_rate * d_w1
            self.b1 = self.b1 - learning_rate * d_b1
            self.w2 = self.w2 - learning_rate * d_w2
            self.b2 = self.b2 - learning_rate * d_b2
        # except Exception as Error:
        #     logger.error(f"Exception error: {Error}.")

    def train(self,TrainInput,TrainTarget):
        if hostname.startswith("rpi"):
            LED_Yellow()
        logger.info("Neural network was started of training.")
        # Генераци весов нейросети по длине входного массива(датасета)
        self.INPUT_DIM = len(TrainInput[0])
        print(self.INPUT_DIM)
        # self.OUTPUT_DIM = TrainTarget.size
        # self.OUTPUT_DIM = len(TrainTarget.tolist()[0])
        print(self.OUTPUT_DIM)
        self.GenerateWeights()
        self.BestModel = False
        # Прохождение по датасету циклом for
        for epoch in track(range(EPOCHS), description='[green]Training model'):
            for TrainInput,TrainTarget in zip(self.batch(TrainInput,BATCH_SIZE),self.batch(TrainTarget,BATCH_SIZE)):
                # Вызов функции для расчёта прямого распространения нейросети
                PredictedValue = self.FeedForward(TrainInput)
                # Вызов функции для расчёта обратного распространения ошибки нейросети
                self.BackwardPropagation(TrainInput,TrainTarget)
                # Расчёт ошибки
                self.Loss = np.sum(self.CrossEntropy(self.OutputLayer, TrainTarget))
                # Сохранение модели с наименьшей ошибкой
                if epoch % (EPOCHS / 20) == 0 and self.Loss <= self.LocalLoss:
                    self.LocalLoss = self.Loss
                    self.save()
                    self.BestModel = True
                # Добавление ошибки в массив для дальнейшего вывода графика ошибки нейросети
                self.LossArray.append(self.Loss)
        # Вывод графика ошибки нейросети
        plt.plot(self.LossArray)
        if self.BestModel == True:
            print("BestModel")
        print(self.Loss)
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

    def predict(self,Input):
        PredictedArray = self.softmax(self.FeedForward(Input))
        print(PredictedArray)
        InputDatasetFile = open("Datasets/SpeechInputDataset.json", "r", encoding ='utf8')
        DataFile = json.load(InputDatasetFile)
        InputDatasetFile.close()
        labelencoder = OneHotEncoder()
        labelencoder.fit_transform(np.array(DataFile).reshape(-1,1))
        PredictedValue = labelencoder.inverse_transform(PredictedArray)
        # Target = labelencoder.inverse_transform([TrainTarget[23]])
        print("Predict")
        print(PredictedValue)
        # print("Target")
        # print(Target)
        return PredictedValue

    # Функция проверки нейросети с последующим выводом количества правильных ответов в виде процентов
    def score(self,TrainInput,TrainTarget):
        InputDatasetFile = open("Datasets/SpeechInputDataset.json", "r", encoding ='utf8')
        DataFile = json.load(InputDatasetFile)
        InputDatasetFile.close()
        vectorizer = LabelEncoder()
        vectorizer.fit_transform(DataFile)
        correct = 0
        id_true = 0
        i = 0
        for Input,Target in zip(TrainInput,TrainTarget):
            i += 1
            PredictedArray = np.argmax(self.FeedForward(Input))
            print(PredictedArray)
            PredictedValue = vectorizer.inverse_transform([PredictedArray])
            Target = vectorizer.inverse_transform([Target])
            if PredictedValue == Target:
                correct += 1
                id_true = i
            print("Target")
            print(Target)
            print("PredictedValue")
            print(PredictedValue)
        accuracy = correct / len(TrainInput)
        print(accuracy)
        print(correct)
        return accuracy

    def load(self,PathParametrs = os.path.join(ProjectDir,'Models','SpeechRecognition.npz')):
        ParametrsFile = np.load(PathParametrs)
        self.GenerateWeights()
        self.w1 = ParametrsFile['arr_0']
        self.w2 = ParametrsFile['arr_1']
        self.b1 = ParametrsFile['arr_2']
        self.b2 = ParametrsFile['arr_3']
        print(ParametrsFile['arr_4'])
        print(ParametrsFile['arr_5'])
        print(ParametrsFile['arr_6'])
        print(ParametrsFile['arr_7'])
        logger.info("Weights of neural network was loaded.")

    def save(self,PathParametrs = os.path.join(ProjectDir,'Models','SpeechRecognition.npz')):
        np.savez_compressed(PathParametrs, self.w1,self.w2,self.b1,self.b2,self.INPUT_DIM,self.HIDDEN_DIM,self.OUTPUT_DIM,learning_rate)

def TestSpeechRecognition():
    speech_recognition = SpeechRecognition()
    # FORMAT = pyaudio.paInt16
    # CHANNELS = 2
    # RATE = 44100
    # CHUNK = 1024
    # RECORD_SECONDS = 2
    # audio = pyaudio.PyAudio()
    # # start Recording
    # stream = audio.open(format=FORMAT, channels=CHANNELS,
    #                 rate=RATE, input=True,
    #                 frames_per_buffer=CHUNK)
    
    # while True:
    #     try:
    #         print ("recording...")
    #         frames = []
            
    #         for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    #             data = stream.read(CHUNK)
    #             frames.append(data)
    #         print ("finished recording")
    #     except KeyboardInterrupt:
    #         # stop Recording
    #         stream.stop_stream()
    #         stream.close()
    #         audio.terminate()
    #         waveFile = wave.open("sound.wave", 'wb')
    #         waveFile.setnchannels(CHANNELS)
    #         waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    #         waveFile.setframerate(RATE)
    #         waveFile.writeframes(b''.join(frames))
    #         waveFile.close()
        
        # print(data)
    speech_recognition.predict(PreprocessingDataset().PreprocessingAudio(PathAudio="sound.wave",mode='predict'))

if __name__ == '__main__':
    TrainInput,TrainTarget = PreprocessingDataset().PreprocessingAudio(PathAudio=AudioDatasetDir,mode='train')
    
    speech_recognition = SpeechRecognition()
    speech_recognition.train(TrainInput,TrainTarget)
    # speech_recognition.load()
    # speech_recognition.predict(TrainInput[23])
    # speech_recognition.predict(TrainInput[23])
    # speech_recognition.score(TrainInput,TrainTarget)
    TestSpeechRecognition()
    # Clean()
    