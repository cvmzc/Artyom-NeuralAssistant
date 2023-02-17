from AssistantCore import Core
import vosk
import sys
import sounddevice as sd
import queue
import json
import asyncio
import os
from loguru import logger
from Services.GetNews import ParseNews
from threading import Thread
import urllib.request
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),"NeuralNetworks"))
from NeuralNetworks.NeuralNetwork import NeuralNetwork
from NeuralNetworks.Preprocessing import PreprocessingDataset

# Инициализация параметров для распознавания речи
SAMPLE_RATE = 16000
DEVICE = 1
# Инициализация параметров для логирования
ProjectDir = os.path.dirname(os.path.realpath(__file__))
logger.add(os.path.join(ProjectDir,'Logs/ArtyomAssistant.log'),format="{time} {level} {message}",level="INFO",rotation="200 MB",diagnose=True)

class ArtyomAssistant():
    def __init__(self):
        super(ArtyomAssistant).__init__()
        if os.path.exists(os.path.join(ProjectDir,'NeuralNetworks','Settings/Settings.json')):
            file = open(os.path.join(ProjectDir,'NeuralNetworks','Settings/Settings.json'),'r',encoding='utf-8')
            DataFile = json.load(file)
            CATEGORIES = DataFile['CATEGORIES']
            CATEGORIES_TARGET = DataFile['CATEGORIES_TARGET']
            file.close()
        else:
            raise RuntimeError
        self.Network = NeuralNetwork(CATEGORIES = CATEGORIES,CATEGORIES_TARGET = CATEGORIES_TARGET)
        self.Preprocessing = PreprocessingDataset()
        self.Network.load()
        self.model = vosk.Model("model_small")
        self.queue = queue.Queue()

    def q_callback(self,indata, frames, time, status):
        # if status:
        #     print(status, file=sys.stderr)
        self.queue.put(bytes(indata))

    def SpeechRecognition(self):
        with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, device=DEVICE, dtype='int16',channels=1, callback=self.q_callback):
            self.Recognizer = vosk.KaldiRecognizer(self.model, SAMPLE_RATE)
            while True:
                data = self.queue.get()
                if self.Recognizer.AcceptWaveform(data):
                    answer = json.loads(self.Recognizer.Result())
                    # print(json.loads(self.Recognizer.Result())["text"])
                    if answer["text"]:
                        yield answer["text"]

    async def Start(self):
        for text in self.SpeechRecognition():
            print(text)
            PreprocessedText = self.Preprocessing.PreprocessingText(PredictArray = [text],mode = 'predict')
            PredictedValue = self.Network.predict(PreprocessedText)
            core = Core()
            await core.CommandManager(PredictedValue)
        # core = Core()
        # await core.CommandManager("Hello")

    
def StartServices():
    try:
        urllib.request.urlopen("http://www.google.com")
        Internet = True
    except IOError:
        Internet = False
    if Internet == True:
        Parse = ParseNews(ProjectDir)
        ParseNewsThread = Thread(target = Parse.StartParse).start()

if __name__ == "__main__":
    # StartServices()
    AsyncioLoop = asyncio.get_event_loop()
    assistant = ArtyomAssistant()
    # assistant.Start()
    AsyncioLoop.run_until_complete(assistant.Start())
