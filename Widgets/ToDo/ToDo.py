import time
import os
from datetime import date
import json
import platform
from loguru import logger
if platform.system() == 'Windows':
    from win10toast import ToastNotifier


class TodoManager:
    def __init__(self,ProjectDir) -> None:
        self.ProjectDir = ProjectDir
        self.UpdateNotes()
        self.LocalDate = date.today().strftime("%B %d, %Y")
        self.LocalTime = (f"{time.strftime('%H')}:{time.strftime('%M')}")
        self.TodoNotes = {"notes":{}}

    def UpdateNotes(self):
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

    def UpdateDate(self):
        self.LocalDate = date.today().strftime("%B %d, %Y")

    def UpdateTime(self):
        self.LocalTime = (f"{time.strftime('%H')}:{time.strftime('%M')}")

    def SaveNotes(self):
        file = open('AssistantConfig/TodoNotes.json','w',encoding='utf-8')
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