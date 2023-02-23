# Импортирование необходмых модулей
import time
import random
import os
import json
import torch
import webbrowser
import sounddevice as sd
import sys
import geocoder
from pyowm import OWM
from Widgets.MusicPlayer.MusicManager import MusicManager
import platform
from Widgets.Timer.Timer import Timer
from Widgets.Stopwatch.Stopwatch import Stopwatch
from NeuralNetworks.CommunicationNetwork import CommunicationNetwork
if platform.system() == 'Windows':
    from win10toast import ToastNotifier
from PIL import ImageGrab
# import asyncio
import wikipedia
import urllib.request
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),"Widgets","CreateProjects"))
from Widgets.CreateProjects.CreateProjects import StartWidget
import threading
import multiprocessing
from Transforms.FilteringTransforms import FilteringTransforms

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
TransformsFile =  open(os.path.join(ProjectDir,'AssistantConfig/Transforms.json'),'r',encoding='utf-8')
Transforms = json.load(TransformsFile)
TransformsFile.close()
NAMES = ['Артём','Артемий','Артёша','Артемьюшка','Артя','Артюня','Артюха','Артюша','Артёмка','Артёмчик','Тёма']
file = open('NeuralNetworks/Settings/ArtyomAnswers.json','r',encoding='utf-8')
ANSWERS  = json.load(file)
file.close()

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
        self.timer = Timer()
        self.stopwatch = Stopwatch()
        self.CommunicationNetwork = CommunicationNetwork()
        self.FilteringTransform = FilteringTransforms()
        ThreadCommunicationFit = threading.Thread(target = self.CommunicationNetwork.Start).start()
    
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
            TransformedText = self.FilteringTransform.to_words(text)
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

    async def CommunicationCommand(self,context:str):
        Answer = self.CommunicationNetwork.predict(context)
        await self.Tell(Answer)

    async def ExitCommand(self):
        await self.Tell(random.choice(ANSWERS['exit']))
    
    async def GratitudeCommand(self):
        await self.Tell(random.choice(ANSWERS['gratitude']))
    
    async def VSCodeCommand(self):
        os.system("code")

    async def YoutubeCommand(self):
        await self.Tell(random.choice(ANSWERS['youtube']))
        webbrowser.open_new_tab('https://youtube.com')
    
    async def WebbrowserCommand(self):
        await self.Tell(random.choice(ANSWERS['webbrowser']))
        webbrowser.open_new_tab('https://google.com')

    # Скриншот
    async def ScreenShotCommand(self):
        image = ImageGrab.grab()
        NameImage = "{}-{}.png".format(time.strftime('%H'),time.strftime('%M'))
        if platform.system() == "Windows":
            ImagePath = os.path.join(os.path.expanduser('~'),'Pictures','Screenshots')
            image.save(NameImage, "PNG")
        elif platform.system() == "Linux":
            pass
        elif platform.system() == "Darwin":
            if os.path.exists(os.path.join(os.path.expanduser('~'),'Desktop','Screenshots')):
                ImagePath = os.path.join(os.path.expanduser('~'),'Desktop','Screenshots')
                image.save(NameImage, "PNG")
            else:
                os.chdir(os.path.expanduser('~'),'Desktop')
                os.mkdir("Screenshots")
                ImagePath = os.path.join(os.path.expanduser('~'),'Desktop','Screenshots')
                image.save(NameImage, "PNG")
    
    # Спящий режим
    async def HibernationCommand(self):
        if platform.system() == 'Windows':
            await self.Tell(random.choice(ANSWERS['hibernation']))
            os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
        elif platform.system() == 'Linux':
            await self.Tell("Эта функция пока не доступна")
        elif platform.system() == 'Darwin':
            await self.Tell("Эта функция пока не доступна")

    # Перезагрузка компьюетра
    async def RebootCommand(self):
        if platform.system() == 'Windows':
            await self.Tell(random.choice(ANSWERS['reboot']))
            os.system("shutdown -t 0 -r -f")
        elif platform.system() == 'Linux':
            await self.Tell("Эта функция пока не доступна")
        elif platform.system() == 'Darwin':
            # os.system("shutdown -r")
            await self.Tell("Эта функция пока не доступна")

    # Выключение компьютера
    async def ShutdownCommand(self):
        if platform.system() == 'Windows':
            await self.Tell(random.choice(ANSWERS['shutdown']))
            os.system('shutdown /p /f')
        elif platform.system() == 'Linux':
            await self.Tell(random.choice(ANSWERS['shutdown']))
            os.system("shutdown -h now")
            # await self.Tell("Эта функция пока не доступна")
        elif platform.system() == 'Darwin':
            await self.Tell(random.choice(ANSWERS['shutdown']))
            os.system("shutdown -h now")
            # await self.Tell("Эта функция пока не доступна")

    # Прогноз погоды
    async def WeatherCommand(self,command:str = "temperature"):
        try:
            urllib.request.urlopen("http://www.google.com")
            self.Internet = True
        except IOError:
            self.Internet = False
            await self.Tell(random.choice(["К сожалению я не могу вам сказать прогноз погоды,так как отсутствует интернет.","Отсутствует подключение к интернету, поэтому я не могу сказать вам прогноз погоды."]))
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
        Answer = wikipedia.summary(text, sentences = 2)
        if len(Answer.split()) > 0:
            await self.Tell(Answer)
        else:
            webbrowser.open(f"https://www.google.com/search?q={text}")
    
    async def News(self):
        file = open("AssistantConfig/News.json","r")
        self.News = json.load(file)
        file.close()
        for news in self.News:
            await self.Tell(news)
    
    async def it_news(self):
        file = open("AssistantConfig/IT_News.json","r")
        self.IT_News = json.load(file)
        file.close()
        for news in self.IT_News:
            await self.Tell(news)

    # Музыкальный плеер
    async def MusicCommand(self,command,text):
        if command == 'music':
            if MusicManager.PausedMusic == False and MusicManager.PlayingMusic == False and MusicManager.StoppedMusic == True:
                MusicThread = threading.Thread(target = MusicManager.PlayMusic)
                MusicThread.start()
            elif MusicManager.PausedMusic == False and MusicManager.PlayingMusic == False and MusicManager.StoppedMusic == False:
                MusicThread = threading.Thread(target = MusicManager.PlayMusic)
                MusicThread.start()
            elif MusicManager.PausedMusic == False and MusicManager.PlayingMusic == True and MusicManager.StoppedMusic == False:
                await self.Tell(random.choice(ANSWERS['play-music']))
            elif MusicManager.PausedMusic == True and MusicManager.PlayingMusic == False and MusicManager.StoppedMusic == False:
                MusicManager.UnpauseMusic()

        elif command == 'off-music':
            if MusicManager.PlayingMusic == True:
                MusicManager.StopMusic()
            elif MusicManager.PlayingMusic == False and MusicManager.StoppedMusic == True:
                await self.Tell(random.choice(ANSWERS['off-music']))

        elif command == 'pause-music':
            if MusicManager.PausedMusic == False and MusicManager.PlayingMusic == True and MusicManager.StoppedMusic == False:
                MusicManager.PauseMusic()
            elif MusicManager.PausedMusic == True and MusicManager.PlayingMusic == False  and MusicManager.StoppedMusic == False:
                await self.Tell(random.choice(ANSWERS['pause-music']))
            elif MusicManager.PausedMusic == False and MusicManager.PlayingMusic == False  and MusicManager.StoppedMusic == True:
                await self.Tell('Музыка выключена.')
                # self.Tell('Включить её?')

        elif command == 'unpause-music':
            if MusicManager.PausedMusic == True and MusicManager.PlayingMusic == False:
                MusicManager.UnpauseMusic()
            elif MusicManager.PausedMusic == False and MusicManager.PlayingMusic == True:
                await self.Tell(random.choice(ANSWERS['unpause-music']))

    async def DjangoProject(self):
        ThreadProject = threading.Thread(target=StartWidget,args=("django_project",))
        ThreadProject.start()

    async def CppProject(self):
        ThreadProject = threading.Thread(target=StartWidget,args=("c++_project",))
        ThreadProject.start()

    async def RustProject(self):
        ThreadProject = threading.Thread(target=StartWidget,args=("rust_project",))
        ThreadProject.start()
    
    async def GoProject(self):
        ThreadProject = threading.Thread(target=StartWidget,args=("go_project",))
        ThreadProject.start()

    async def NodeJSProject(self):
        ThreadProject = threading.Thread(target=StartWidget,args=("NodeJS_project",))
        ThreadProject.start()

    async def CProject(self):
        ThreadProject = threading.Thread(target=StartWidget,args=("c_project",))
        ThreadProject.start()

    async def JavaProject(self):
        ThreadProject = threading.Thread(target=StartWidget,args=("java_project",))
        ThreadProject.start()
    
    async def PythonProject(self):
        ThreadProject = threading.Thread(target=StartWidget,args=("python_project",))
        ThreadProject.start()
        # StartWidget("python_project")

    async def CreateEchoBot(self):
        pass

    async def CommandManager(self,PredictedValue):
        print(PredictedValue)
        # await self.Tell("Привет")
        # await self.WeatherCommand()
        # await self.PythonProject()
        # await self.Functions["weather"]()
        # await self.Functions[PredictedValue]()

# if __name__ == "__main__":
#     AsyncioLoop = asyncio.get_event_loop()
#     core = Core()
#     AsyncioLoop.run_until_complete(core.CommandManager("communication"))
