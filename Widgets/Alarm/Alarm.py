import time
from datetime import date
import sounddevice as sd
import soundfile as sf
from loguru import logger
import os
import json
from plyer import notification 
# import asyncio

class Alarm:
    def __init__(self,ProjectDir):
        global AudioPath
        global DefaultAudioPath
        self.ProjectDir = ProjectDir
        DefaultAudioPath = os.path.join(ProjectDir,"Audio/Alarm.wav")
        AudioPath = DefaultAudioPath
        self.Alarms = {"alarms":{}}
        self.CheckAlarmFile()

    def CheckAlarmFile(self):
        if os.path.exists(os.path.join(self.ProjectDir,'AssistantConfig/Alarm.json')):
            file = open(os.path.join(self.ProjectDir,'AssistantConfig/Alarm.json'),'r',encoding='utf-8')
            self.Alarms = json.load(file)
            file.close()
        else:
            file = open(os.path.join(self.ProjectDir,'AssistantConfig/Alarm.json'),"w")
            json.dump(self.Alarms,file,ensure_ascii=False,sort_keys=True, indent=2)
            file.close()

    async def Notification(self,title,message):
        notification.notify(  
            title = title,  
            message = message,  
            app_icon = None,  
            timeout = 10,  
            toast = False  
        ) 

    async def UpdateDate(self):
        self.LocalDate = date.today().strftime("%B %d, %Y")

    async def UpdateTime(self):
        self.LocalTime = str(time.strftime("%H") + ":" + time.strftime("%M"))

    async def UpdateAlarm(self):
        file = open(os.path.join(self.ProjectDir,'AssistantConfig/Alarm.json'),'r',encoding='utf-8')
        self.Alarms = json.load(file)
        file.close()
    
    async def SaveAlarms(self):
        file = open(os.path.join(self.ProjectDir,'AssistantConfig/Alarm.json'),"w")
        json.dump(self.Alarms,file,ensure_ascii=False,sort_keys=True, indent=2)
        file.close()

    async def CreateAlarm(self,date:str = date.today().strftime("%B %d, %Y"),time:str = str(time.strftime("%H") + ":" + time.strftime("%M"))):
        await self.UpdateAlarm()
        await self.UpdateDate()
        await self.UpdateTime()
        if date in self.Alarms["alarms"]:
            if not time in self.Alarms["alarms"][date]:
                self.Alarms["alarms"][date].append(time)
        else:
            self.Alarms["alarms"].update(
                {
                            date:[time]
                }
            )
        await self.SaveAlarms()

    async def RemoveAlarm(self,date,time):
        await self.UpdateAlarm()
        if date in self.Alarms["alarms"]:
            if time in self.Alarms["alarms"][date]:
                if len(self.Alarms["alarms"][date]) >= 2:
                    self.Alarms["alarms"][date].remove(time)
                    await self.SaveAlarms()
                elif len(self.Alarms["alarms"][date]) == 1:
                    self.Alarms["alarms"].pop(date)
                    await self.SaveAlarms()

    async def RingAlarm(self):
        AudioData,sample_rate = sf.read(AudioPath,dtype = 'float32')
        sd.play(AudioData,sample_rate)
        StatusAudio = sd.wait()

    async def CheckAlarm(self):
        await self.UpdateAlarm()
        while True:
            await self.UpdateDate()
            await self.UpdateTime()
            if self.LocalDate in self.Alarms["alarms"]:
                for TimeAlarm in self.Alarms["alarms"][self.LocalDate]:
                    if TimeAlarm == self.LocalTime:
                        await self.Notification(title = "Будильник",message = self.LocalTime)
                        await self.RingAlarm()
                        await self.RemoveAlarm(self.LocalDate,self.LocalTime)
                        

# if __name__ == "__main__":
#     AsyncioLoop = asyncio.get_event_loop()
#     alarm = Alarm(os.getcwd())
#     AsyncioLoop.run_until_complete(alarm.CreateAlarm(time = "23:11"))