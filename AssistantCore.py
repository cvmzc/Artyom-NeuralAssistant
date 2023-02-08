# Импортирование необходмых модулей
import time
import random
import os
import json
import torch
import webbrowser
import sounddevice as sd
import sys
from Preprocessing import PreprocessingDataset
from NeuralNetwork import NeuralNetwork
import geocoder
from pyowm import OWM
import threading
from MusicManager import MusicManager
import platform
from loguru import logger
from datetime import date
from Alarm import Alarm
from Widgets.Timer.Timer import Timer
from Widgets.Stopwatch.Stopwatch import Stopwatch
from CommunicationNetwork import CommunicationNetwork
if platform.system() == 'Windows':
    from win10toast import ToastNotifier
from PIL import ImageGrab
import asyncio
import wikipedia
import urllib.request
from Widgets import CreateProjects
# Инициализация параметров
ProjectDir = os.path.dirname(os.path.realpath(__file__))
UserDir = os.path.expanduser('~')
wikipedia.set_lang("ru")

#TTS параметры
LANGUAGE = 'ru'
MODEL_ID = 'ru_v3'
SAMPLE_RATE = 48000 # 48000
SPEAKER = 'aidar' # aidar, baya, kseniya, xenia, random
put_accent = True
put_yo = True
DEVICE = torch.device('cpu')

# Импортирование необходмых для полноценной работы ассистента классов
TransformsFile =  open(os.path.join(ProjectDir,'AssistantSettings/Transforms.json'),'r',encoding='utf-8')
Transforms = json.load(TransformsFile)
TransformsFile.close()
NAMES = ['Артём','Артемий','Артёша','Артемьюшка','Артя','Артюня','Артюха','Артюша','Артёмка','Артёмчик','Тёма']

class Core:
    def __init__(self):
        self.owm = OWM('2221d769ed67828e858caaa3803161ea')
        self.Functions = {
            'communication':self.CommunicationCommand,'weather':self.WeatherCommand,
            "c++_project":self.CppProject,"django_project":self.DjangoProject,"python_project":self.PythonProject,"NodeJS_project":self.NodeJSProject,
            "rust_project":self.RustProject,"c_project":self.CProject,"go_project":self.GoProject,"java_project":self.JavaProject
        }
        self.model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                          model='silero_tts',
                          language=LANGUAGE,
                          speaker=MODEL_ID)
        self.model.to(DEVICE)
    
    async def Tell(self,text):
        audio = self.model.apply_tts(text=text+"..",
                            speaker=SPEAKER,
                            sample_rate=SAMPLE_RATE,
                            put_accent=put_accent,
                            put_yo=put_yo)

        sd.play(audio, SAMPLE_RATE * 1.05)
        time.sleep((len(audio) / SAMPLE_RATE) + 0.5)
        sd.stop()
    
    async def FilteringTransforms(self,text:str,to_nums:bool = False,to_words:bool = False,from_date:bool = False):
        if to_nums == True:
            TransformedText = ""
            Nums = []
            LocalText =  text.split()
            for word in LocalText:
                if word in Transforms["Words"]:
                    Nums.append(int(Transforms["Words"][word]))
                    text = text.replace(word,str(Transforms["Words"][word]))
            TransformedText = text
            return TransformedText,Nums
        elif to_words == True:
            TransformedText = ""
            Words = []
            LocalText = text.split()
            for number in LocalText:
                if number in Transforms["Nums"]:
                    Words.append(Transforms["Nums"][number])
                    text = text.replace(number,str(Transforms["Nums"][number]))
            TransformedText = text
            return TransformedText
        elif from_date == True:
            TransformedText = ""
            Words = []
            LocalText = text.split()
            for string_date in LocalText:
                if string_date in Transforms["Date"]:
                    Words.append(Transforms["Date"][string_date])
                    text = text.replace(string_date,str(Transforms["Date"][string_date]))
            TransformedText = text
            return TransformedText

    async def CommunicationCommand(self):
        print("hello")
    
    async def WeatherCommand(self,command:str = "temperature"):
        try:
            urllib.request.urlopen("http://www.google.com")
            self.Internet = True
        except IOError:
            self.Internet = False
            self.Tell(random.choice(["К сожалению я не могу вам сказать прогноз погоды,так как отсутствует интернет.","Отсутствует подключение к интернету, поэтому я не могу сказать вам прогноз погоды."]))
        if self.Internet == True:
            geolocation = geocoder.ip('me')
            coordinates = geolocation.latlng
            mgr = self.owm.geocoding_manager()
            city = mgr.reverse_geocode(lat=coordinates[0], lon=coordinates[1])
            mgr = self.owm.weather_manager()
            observation = mgr.weather_at_place(city[0].name)  # the observation object is a box containing a weather object
            weather = observation.weather
            if command == 'temperature':
                temp = int(weather.temperature('celsius')["temp"])
                wind = weather.wind()["speed"]
                wind = f"{wind} метров в секунду"
                humidity = str(weather.humidity) + " процентов"
                temp_str = await self.FilteringTransforms(f'Сейчас {temp} градуса',to_words=True)
                print(temp_str)
                await self.Tell(str(temp_str))
    async def WikiCommand(self,text):
        wikipedia.summary(text)

    async def DjangoProject(self):
        ThreadProject = threading.Thread(target=CreateProjects.StartProject,args=("django_project"))
        ThreadProject.start()

    async def CppProject(self):
        ThreadProject = threading.Thread(target=CreateProjects.StartProject,args=("c++_project"))
        ThreadProject.start()

    async def RustProject(self):
        ThreadProject = threading.Thread(target=CreateProjects.StartProject,args=("rust_project"))
        ThreadProject.start()
    
    async def GoProject(self):
        ThreadProject = threading.Thread(target=CreateProjects.StartProject,args=("go_project"))
        ThreadProject.start()

    async def NodeJSProject(self):
        ThreadProject = threading.Thread(target=CreateProjects.StartProject,args=("NodeJS_project"))
        ThreadProject.start()

    async def CProject(self):
        ThreadProject = threading.Thread(target=CreateProjects.StartProject,args=("c_project"))
        ThreadProject.start()

    async def JavaProject(self):
        ThreadProject = threading.Thread(target=CreateProjects.StartProject,args=("java_project"))
        ThreadProject.start()
    
    async def PythonProject(self):
        ThreadProject = threading.Thread(target=CreateProjects.StartProject,args=("python_project"))
        ThreadProject.start()

    async def CreateEchoBot(self):
        pass

    async def CommandManager(self,PredictedValue):
        # await self.Tell("Привет")
        await self.WeatherCommand()
        # await self.Functions["weather"]()
        # await self.Functions[PredictedValue]()

if __name__ == "__main__":
    AsyncioLoop = asyncio.get_event_loop()
    core = Core()
    AsyncioLoop.run_until_complete(core.CommandManager("communication"))
