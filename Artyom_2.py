# Импортирование необходмых модулей
import time
import random
import os
import json
import torch
import webbrowser
import sounddevice as sd
import sys
import pyaudio
from vosk import Model, KaldiRecognizer
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
from Timer.Timer import Timer
from Stopwatch.Stopwatch import Stopwatch
if platform.system() == 'Windows':
    from win10toast import ToastNotifier
from PIL import ImageGrab

# Инициализация параметров
ProjectDir = os.path.dirname(os.path.realpath(__file__))
UserDir = os.path.expanduser('~')
file = open(os.path.join(ProjectDir,'NeuralNetworkSettings/Settings.json'),'r',encoding='utf-8')
DataFile = json.load(file)
CATEGORIES = DataFile['CATEGORIES']
CATEGORIES_TARGET = DataFile['CATEGORIES_TARGET']
file.close()
NAMES = ['Артём','Артемий','Артёша','Артемьюшка','Артя','Артюня','Артюха','Артюша','Артёмка','Артёмчик','Тёма']
file = open(os.path.join(ProjectDir,'NeuralNetworkSettings/ArtyomAnswers.json'),'r',encoding='utf-8')
ANSWERS  = json.load(file)
file.close()
language = 'ru'
model_id = 'ru_v3'
sample_rate = 48000 # 48000
speaker = 'aidar' # aidar, baya, kseniya, xenia, random
put_accent = True
put_yo = True
device = torch.device('cpu') # cpu или gpu
MusicManager = MusicManager()
TransformsFile =  open(os.path.join(ProjectDir,'AssistantSettings/Transforms.json'),'r',encoding='utf-8')
Transforms = json.load(TransformsFile)
TransformsFile.close()
Preprocessing = PreprocessingDataset() 
network = NeuralNetwork(CATEGORIES,CATEGORIES_TARGET)
network.load()
owm = OWM('2221d769ed67828e858caaa3803161ea')
logger.add(os.path.join(ProjectDir,'Logs/ArtyomAssistant.log'),format="{time} {level} {message}",level="INFO",rotation="200 MB",diagnose=True)
timer = Timer()
stopwatch = Stopwatch()

class ArtyomAssistant:
    def __init__(self):
        # self.Functions = [
        #     self.CommunicationCommand,self.WeatherCommand,self.YoutubeCommand,
        #     self.WebbrowserCommand,self.MusicCommand,self.NewsCommand,
        #     self.ToDoCommand,self.CalendarCommand,self.JoikesCommand,
        #     self.ExitCommand,self.TimeCommand,self.GratitudeCommand,
        #     self.StopwatchCommand,self.TimerCommand,self.ShutdownCommand,
        #     self.RebootCommand,self.HibernationCommand,self.AlarmCommand,
        #     self.FavouriteAppCommand,self.VSCodeCommand,self.NotepadCommand,
        #     self.MailCommand,self.DateCommand,self.HowAreYouCommand,self.WhatYouDoCommand
        # ]
        self.Functions = {
            'communication':self.CommunicationCommand,'weather':self.WeatherCommand,
            'time':self.TimeCommand,'youtube':self.YoutubeCommand,
            'webbrowser':self.WebbrowserCommand,'hibernation':self.HibernationCommand,'reboot':self.RebootCommand,
            'shutdown':self.ShutdownCommand,'news':self.NewsCommand,
            'todo':self.TodoCommand,'calendar':self.CalendarCommand,
            'joikes':self.JoikesCommand,'exit':self.ExitCommand,
            'gratitude':self.GratitudeCommand,'vscode':self.VSCodeCommand,
            'todo':self.ToDoCommand,'alarm':self.AlarmCommand,
            'timer':self.TimerCommand,'stopwatch':self.StopwatchCommand,
            'screenshot':self.ScreenShotCommand,
        }
        self.RecognitionModel = Model('model')
        self.Recognition = KaldiRecognizer(self.RecognitionModel,16000)
        self.RecognitionAudio = pyaudio.PyAudio()
        self.stream = self.RecognitionAudio.open(format=pyaudio.paInt16,channels=1,rate=16000,input=True,frames_per_buffer=8000)
        self.stream.start_stream()

        self.model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                          model='silero_tts',
                          language=language,
                          speaker=model_id)
        self.model.to(device)

    def Tell(self,text: str):
        audio = self.model.apply_tts(text=text+"..",
                                speaker=speaker,
                                sample_rate=sample_rate,
                                put_accent=put_accent,
                                put_yo=put_yo)

        sd.play(audio, sample_rate * 1.05)
        time.sleep((len(audio) / sample_rate) + 0.5)
        sd.stop()

    def SpeechRecognition(self):
        while True:
            data = self.stream.read(8000,exception_on_overflow=False)
            if (self.Recognition.AcceptWaveform(data)) and (len(data) > 0):
                answer = json.loads(self.Recognition.Result())
                if answer['text']:
                    yield answer['text']
    
    def FilteringTransforms(self,text:str,to_nums:bool = True,to_words:bool = False,from_date:bool = False):
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
            print(TransformedText)
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
    
    def CommunicationCommand(self):
        self.Tell(random.choice(ANSWERS['communication']))
    
    def FavouriteAppCommand(self):
        pass

    def NotepadCommand(self):
        pass

    def MailCommand(self):
        pass

    def HowAreYouCommand(self):
        pass

    def WhatYouDoCommand(self):
        pass

    def WeatherCommand(self,WithInternet:bool=False):
        geolocation = geocoder.ip('me')
        coordinates = geolocation.latlng
        mgr = owm.geocoding_manager()
        city = mgr.reverse_geocode(lat=coordinates[0], lon=coordinates[1])

        mgr = owm.weather_manager()
        observation = mgr.weather_at_place(city[0].name)  # the observation object is a box containing a weather object
        weather = observation.weather
        temp = int(weather.temperature('celsius')["temp"])
        wind = weather.wind()["speed"]
        wind = f"{wind} метров в секунду"
        humidity = str(weather.humidity) + " процентов"
        temp_str = self.FilteringTransforms(f'Сейчас {temp} градусов',to_words=True)
        self.Tell(temp_str)
    
    def DateCommand(self):
        LocalDate = date.today().strftime("%d %B")
        LocalDate = self.FilteringTransforms(LocalDate)
        LocalDate_str = f"Сегодня {LocalDate}"
        self.Tell(LocalDate_str)
    
    def TimeCommand(self):
        hours = int(time.strftime('%H'))
        minutes = int(time.strftime('%M'))
        time_str = self.FilteringTransforms(f'Сейчас {hours} {minutes}',to_words=True)
        print(time_str)
        self.Tell(time_str)

    def MusicCommand(self,command,text):
        if command == 'music':
            if MusicManager.PausedMusic == False and MusicManager.PlayingMusic == False and MusicManager.StoppedMusic == True:
                MusicThread = threading.Thread(target = MusicManager.PlayMusic)
                MusicThread.start()
            elif MusicManager.PausedMusic == False and MusicManager.PlayingMusic == False and MusicManager.StoppedMusic == False:
                MusicThread = threading.Thread(target = MusicManager.PlayMusic)
                MusicThread.start()
            elif MusicManager.PausedMusic == False and MusicManager.PlayingMusic == True and MusicManager.StoppedMusic == False:
                self.Tell(random.choice(ANSWERS['play-music']))
            elif MusicManager.PausedMusic == True and MusicManager.PlayingMusic == False and MusicManager.StoppedMusic == False:
                MusicManager.UnpauseMusic()

        elif command == 'off-music':
            if MusicManager.PlayingMusic == True:
                MusicManager.StopMusic()
            elif MusicManager.PlayingMusic == False and MusicManager.StoppedMusic == True:
                self.Tell(random.choice(ANSWERS['off-music']))

        elif command == 'pause-music':
            if MusicManager.PausedMusic == False and MusicManager.PlayingMusic == True and MusicManager.StoppedMusic == False:
                MusicManager.PauseMusic()
            elif MusicManager.PausedMusic == True and MusicManager.PlayingMusic == False  and MusicManager.StoppedMusic == False:
                self.Tell(random.choice(ANSWERS['pause-music']))
            elif MusicManager.PausedMusic == False and MusicManager.PlayingMusic == False  and MusicManager.StoppedMusic == True:
                self.Tell('Музыка выключена.')
                # self.Tell('Включить её?')

        elif command == 'unpause-music':
            if MusicManager.PausedMusic == True and MusicManager.PlayingMusic == False:
                MusicManager.UnpauseMusic()
            elif MusicManager.PausedMusic == False and MusicManager.PlayingMusic == True:
                self.Tell(random.choice(ANSWERS['unpause-music']))

    def YoutubeCommand(self):
        self.Tell(random.choice(ANSWERS['youtube']))
        webbrowser.open_new_tab('https://youtube.com')
    
    def WebbrowserCommand(self):
        self.Tell(random.choice(ANSWERS['webbrowser']))
        webbrowser.open_new_tab('https://google.com')

    def HibernationCommand(self):
        if platform.system() == 'Windows':
            self.Tell(random.choice(ANSWERS['hibernation']))
            os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
        elif platform.system() == 'Linux':
            self.Tell("Эта функция пока не доступна")
        elif platform.system() == 'Darwin':
            self.Tell("Эта функция пока не доступна")

    def RebootCommand(self):
        if platform.system() == 'Windows':
            self.Tell(random.choice(ANSWERS['reboot']))
            os.system("shutdown -t 0 -r -f")
        elif platform.system() == 'Linux':
            self.Tell("Эта функция пока не доступна")
        elif platform.system() == 'Darwin':
            self.Tell("Эта функция пока не доступна")

    def ShutdownCommand(self):
        if platform.system() == 'Windows':
            self.Tell(random.choice(ANSWERS['shutdown']))
            os.system('shutdown /p /f')
        elif platform.system() == 'Linux':
            self.Tell("Эта функция пока не доступна")
        elif platform.system() == 'Darwin':
            self.Tell("Эта функция пока не доступна")

    def NewsCommand(self):
        pass

    def TodoCommand(self):
        self.Tell("Эта функция пока не доступна")

    def CalendarCommand(self):
        self.Tell("Эта функция пока не доступна")

    def JoikesCommand(self):
        pass

    def ExitCommand(self):
        self.Tell(random.choice(ANSWERS['exit']))

    def GratitudeCommand(self):
        self.Tell(random.choice(ANSWERS['gratitude']))

    def ToDoCommand(self,command,text:None):
        if command == 'todo':
            pass

    def VSCodeCommand(self):
        if platform.system() == 'Windows':
            if os.path.exists(os.path.join(UserDir,'/AppData/Local/Programs/Microsoft VS Code/Code.exe')):
                self.Tell(random.choice(ANSWERS['vscode']))
                os.startfile(os.path.join(UserDir,'/AppData/Local/Programs/Microsoft VS Code/Code.exe'))
            else:
                self.Tell(random.choice(['У вас не установлена эта программа','Редактор кода не установлен на этом компьютере','Программа не установлена на этом компьютере']))
        elif platform.system() == 'Linux':
            self.Tell("Эта функция пока не доступна")
        elif platform.system() == 'Darwin':
            self.Tell("Эта функция пока не доступна")

    def TimerCommand(self,command,text:None):
        hours = 0
        minutes = 0
        seconds = 0
        TransformedText,Nums = self.FilteringTransforms(text,to_nums = True)
        if command == 'timer':
            timer = Timer()
            if len(Nums) == 3:
                hours = Nums[0]
                minutes = Nums[1]
                seconds = Nums[2]
            elif len(Nums) == 2:
                hours = 0
                minutes = Nums[0]
                seconds = Nums[1]
            elif len(Nums) == 1:
                hours = 0
                minutes = 0
                seconds = Nums[0]
            timer.Main(hours,minutes,seconds)
            if timer.TimeStarted == False  and timer.TimeUnpaused == False and timer.TimePaused == False:
                self.Tell("Таймер запущен")
            else:
                self.Tell("Таймер не удалось запустить")

        elif command == 'off-timer':
            Progress = timer.Stop()
            if Progress == True:
                self.Tell(random.choice(["Таймер выключен","Выключил"]))
            elif Progress == False:
                self.Tell("Таймер не запущен")

        elif command == 'pause-timer':
            Progress = timer.Pause()
            if Progress == True:
                self.Tell(random.choice(["Таймер поставлен на паузу","Таймер остановлен","Остановил"]))

        elif command == 'unpause-timer':
            Progress = timer.Unpause()
            if Progress == True:
                self.Tell(random.choice(["Таймер возобновлён","Отсчёт продолжается","Таймер "]))
            

    def StopwatchCommand(self,command,text:None):
        stopwatch = Stopwatch()
        if command == 'stopwatch':
            stopwatch.Main()
        elif command == 'off-stopwatch':
            stopwatch.Stop()
        elif command == 'pause-stopwatch':
            stopwatch.Pause()
        elif command == 'unpause-stopwatch':
            stopwatch.Unpause()
    
    def ScreenShotCommand(self):
        image = ImageGrab()
        NameImage = "{}-{}.png".format(time.strftime('%H'),time.strftime('%M'))
        if platform.system() == "Windows":
            ImagePath = os.path.join(os.path.expanduser('~'),'Pictures','Screenshots')
            image.save(NameImage, "PNG")
        elif platform.system() == "Linux":
            pass
        elif platform.system() == "Darwin":
            pass

    def AlarmCommand(self):
        pass

    def CommandManager(self,PredictedValue,text:None,PredictedInt:int):
        # self.Functions[PredictedInt]
        if PredictedValue == "don't_know":
            self.Tell(random.choice(ANSWERS["don't_know"]))
        else:
            operation = PredictedValue#CATEGORIES[PredictedValue]

            if operation == 'music' or operation == 'off-music' or operation == 'pause-music' or operation == 'unpause-music':
                self.MusicCommand(operation,text)
            elif operation == 'stopwatch' or operation == 'off-stopwatch' or operation == 'pause-stopwatch' or 'unpause-stopwatch':
                self.StopwatchCommand(operation,text)
            elif operation == 'timer' or operation == 'off-timer' or operation == 'pause-timer' or operation == 'unpause-timer':
                self.TimerCommand(operation,text)
            elif operation == 'time':
                self.TimeCommand()
            else:
                print("Hello")
                self.Functions[PredictedValue]
                self.WeatherCommand()

    def Start(self):
        # self.Alarm_Class = Alarm()
        # AlarmThread = threading.Thread(target = self.Alarm_Class.CheckAlarm)
        # AlarmThread.start()
        # self.ToDo = TodoManager()
        # ToDoThread = threading.Thread(target = self.ToDo.CheckNote)
        # ToDoThread.start()
        for text in self.SpeechRecognition():
            print(text)
            for name in NAMES:
                if name.lower() in text and len(text.split()) > 1:
                    Input = text.replace(name.lower(),"")
                    Input = [text]
                    Input = Preprocessing.PreprocessingText(PredictArray = Input,mode = 'predict')
                    PredictedValue,PredictedInt = network.predict(Input)
                    self.WeatherCommand()
                    self.CommandManager(PredictedValue,text,PredictedInt)
                    break
                elif name.lower() in text and len(text.split()) == 1:
                    self.Tell('Чем могу помочь?')
                    break
                
class TodoManager(ArtyomAssistant):
    def __init__(self) -> None:
        self.UpdateNotes()
        self.LocalDate = date.today().strftime("%B %d, %Y")
        self.LocalTime = (f"{time.strftime('%H')}:{time.strftime('%M')}")
        self.TodoNotes = {"notes":{}}

    def UpdateNotes(self):
        if os.path.exists(os.path.join(ProjectDir,'AssistantSettings/TodoNotes.json')):
            file = open('AssistantSettings/TodoNotes.json','r',encoding='utf-8')
            DataFile = json.load(file)
            self.TodoNotes = DataFile
            file.close()
        else:
            file = open('AssistantSettings/TodoNotes.json','w',encoding='utf-8')
            json.dump({"notes":{}},file,ensure_ascii=False,sort_keys=True, indent=2)
            file.close()
            file = open('AssistantSettings/TodoNotes.json','r',encoding='utf-8')
            DataFile = json.load(file)
            self.TodoNotes = DataFile
            file.close()

    def UpdateDate(self):
        self.LocalDate = date.today().strftime("%B %d, %Y")

    def UpdateTime(self):
        self.LocalTime = (f"{time.strftime('%H')}:{time.strftime('%M')}")

    def SaveNotes(self):
        file = open('AssistantSettings/TodoNotes.json','w',encoding='utf-8')
        json.dump(self.TodoNotes,file,ensure_ascii=False,sort_keys=True, indent=2)
        file.close()

    def Notification(self,title,message):
        if platform.system() == 'Windows':
            ToastNotifier().show_toast(title=title,msg=message,duration=5)

    def CreateNote(self,text:str,date:str = date.today().strftime("%B %d, %Y"),time:str = (f"{time.strftime('%H')}:{time.strftime('%M')}")):
        self.UpdateNotes()
        # print(self.TodoNotes)
        if date in self.TodoNotes["notes"]:
            if time in self.TodoNotes["notes"][date]:
                if not text in self.TodoNotes["notes"][date][time]:
                    self.TodoNotes["notes"][date][time].append(text)
                    print(self.TodoNotes)
            else:
                self.TodoNotes["notes"][date].update({time:[text]})
        else:
            self.TodoNotes["notes"].update(
                {
                            date:{
                                    time:[text]
                                }
                }
            )
        self.SaveNotes()

    def RemoveNote(self,text:str,date:str = date.today().strftime("%B %d, %Y"),time:str = (f"{time.strftime('%H')}:{time.strftime('%M')}")):
        self.UpdateNotes()
        if date in self.TodoNotes["notes"]:
            if time in self.TodoNotes["notes"][date]:
                if text in self.TodoNotes["notes"][date][time]:
                    self.TodoNotes["notes"][date][time].remove(text)
                    self.SaveNotes()
        # else:
        #     self.Tell(random.choice(["Заметка отсутствует","Я не нашёл такой заметки","Заметка не найдена","Такой заметки нет","Такой заметки не существует"]))
        

    def CheckNote(self):
        self.UpdateNotes()
        while True:
            self.UpdateDate()
            self.UpdateTime()
            if self.LocalDate in self.TodoNotes["notes"]:
                if self.LocalTime in self.TodoNotes["notes"][self.LocalDate]:
                    for note in self.TodoNotes["notes"][self.LocalDate][self.LocalTime]:
                        self.Notification("Заметка",note)
                        if len(self.TodoNotes["notes"][self.LocalDate][self.LocalTime]) >= 2:
                            self.TodoNotes["notes"][self.LocalDate][self.LocalTime].remove(note)
                            self.SaveNotes()
                        elif len(self.TodoNotes["notes"][self.LocalDate][self.LocalTime]) == 1:
                            self.TodoNotes["notes"][self.LocalDate].pop(self.LocalTime)
                            self.SaveNotes()



if __name__ == '__main__':
    # todo_manager = TodoManager()
    # todo_manager.CreateNote("123456789",date = todo_manager.DefaultDate,self.LocalTime = "23:31")
    # todo_manager.CreateNote("Hello,BRO :)",date = todo_manager.DefaultDate,self.LocalTime = "23:32")
    # todo_manager.CheckNote()
    Artyom = ArtyomAssistant()
    Artyom.Start()
