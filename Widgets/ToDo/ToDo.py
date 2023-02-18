import time
import os
from datetime import date
import json
from loguru import logger
from plyer import notification 

class TodoManager:
    def __init__(self,ProjectDir) -> None:
        self.ProjectDir = ProjectDir
        self.UpdateNotes()
        self.LocalDate = date.today().strftime("%B %d, %Y")
        self.LocalTime = (f"{time.strftime('%H')}:{time.strftime('%M')}")
        self.TodoNotes = {"notes":{}}

    async def UpdateNotes(self):
        if os.path.exists(os.path.join(self.ProjectDir,'AssistantConfig/TodoNotes.json')):
            file = open('AssistantConfig/TodoNotes.json','r',encoding='utf-8')
            DataFile = json.load(file)
            self.TodoNotes = DataFile
            file.close()
        else:
            file = open('AssistantConfig/TodoNotes.json','w',encoding='utf-8')
            json.dump({"notes":{}},file,ensure_ascii=False,sort_keys=True, indent=2)
            file.close()
            file = open('AssistantConfig/TodoNotes.json','r',encoding='utf-8')
            DataFile = json.load(file)
            self.TodoNotes = DataFile
            file.close()

    async def UpdateDate(self):
        self.LocalDate = date.today().strftime("%B %d, %Y")

    async def UpdateTime(self):
        self.LocalTime = (f"{time.strftime('%H')}:{time.strftime('%M')}")

    async def SaveNotes(self):
        file = open('AssistantConfig/TodoNotes.json','w',encoding='utf-8')
        json.dump(self.TodoNotes,file,ensure_ascii=False,sort_keys=True, indent=2)
        file.close()

    async def Notification(self,title,message):
        notification.notify(  
            title = title,  
            message = message,  
            app_icon = None,  
            timeout = 10,  
            toast = False  
        ) 

    async def CreateNote(self,text:str,date:str = date.today().strftime("%B %d, %Y"),time:str = (f"{time.strftime('%H')}:{time.strftime('%M')}")):
        await self.UpdateNotes()
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
        await self.SaveNotes()

    async def RemoveNote(self,text:str,date:str = date.today().strftime("%B %d, %Y"),time:str = (f"{time.strftime('%H')}:{time.strftime('%M')}")):
        await self.UpdateNotes()
        if date in self.TodoNotes["notes"]:
            if time in self.TodoNotes["notes"][date]:
                if text in self.TodoNotes["notes"][date][time]:
                    self.TodoNotes["notes"][date][time].remove(text)
                    await self.SaveNotes()
        # else:
        #     self.Tell(random.choice(["Заметка отсутствует","Я не нашёл такой заметки","Заметка не найдена","Такой заметки нет","Такой заметки не существует"]))
        

    async def CheckNote(self):
        await self.UpdateNotes()
        while True:
            await self.UpdateDate()
            await self.UpdateTime()
            if self.LocalDate in self.TodoNotes["notes"]:
                if self.LocalTime in self.TodoNotes["notes"][self.LocalDate]:
                    for note in self.TodoNotes["notes"][self.LocalDate][self.LocalTime]:
                        await self.Notification("Заметка",note)
                        if len(self.TodoNotes["notes"][self.LocalDate][self.LocalTime]) >= 2:
                            self.TodoNotes["notes"][self.LocalDate][self.LocalTime].remove(note)
                            await self.SaveNotes()
                        elif len(self.TodoNotes["notes"][self.LocalDate][self.LocalTime]) == 1:
                            self.TodoNotes["notes"][self.LocalDate].pop(self.LocalTime)
                            await self.SaveNotes()