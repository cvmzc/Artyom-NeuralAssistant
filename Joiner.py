import json
import os

# Инициализация переменных
symbols = ['"','?','!','/']
Limit = 3500

# Подготовка датасета
ProjectDir = os.getcwd()
DatasetFile = open(os.path.join(ProjectDir,"Datasets/ArtyomDataset.json"),"r",encoding="utf-8")
Dataset = json.load(DatasetFile)
DatasetFile.close

WikiDatasetFile = open(os.path.join(ProjectDir,"Datasets/WikiDataset.json"),"r",encoding="utf-8")
WikiDataset = json.load(WikiDatasetFile)
WikiDatasetFile.close

WikiSettingsFile = open(os.path.join(ProjectDir,"NeuralNetworkSettings/WikiSettings.json"),"r",encoding="utf-8")
WikiSettings = json.load(WikiSettingsFile)
WikiSettingsFile.close()

AdditionalDatasetFile = open(os.path.join(ProjectDir,"Datasets/RuBQ_2.0.json"),"r",encoding="utf-8")
AdditionalDataset = json.load(AdditionalDatasetFile)
AdditionalDatasetFile.close()

AdditionalDatasetFile_2 = open(os.path.join(ProjectDir,"Datasets/RuBQ_2.0_test.json"),"r",encoding="utf-8")
AdditionalDataset_2 = json.load(AdditionalDatasetFile_2)
AdditionalDatasetFile_2.close()

ArtyomSettingsFile = open(os.path.join(ProjectDir,"NeuralNetworkSettings/ArtyomAnswers.json"),"r",encoding="utf-8")
ArtyomSettings = json.load(ArtyomSettingsFile)
ArtyomSettingsFile.close()

SettingsFile = open(os.path.join(ProjectDir,"NeuralNetworkSettings/Settings.json"),"r",encoding="utf-8")
Settings = json.load(SettingsFile)
SettingsFile.close()

def RuBQ_2():
    LatestInt = -1
    for value in WikiSettings["CATEGORIES"]:
        LatestInt += 1
    print(LatestInt)
    for Group in AdditionalDataset[:Limit]:
        # for symbol in symbols:
        Question = (Group["question_text"]).replace('"',"")
        Answer = (Group["answer_text"]).replace('"',"")
        WikiDataset["dataset"].update(
            {
                Answer:{
                    "questions": [Question]
                }
            }
        )
        # LatestInt_2 = LatestInt
        LatestInt += 1
        WikiSettings["CATEGORIES"].update(
            {
                Answer: LatestInt
            }
        )

        WikiSettings["CATEGORIES_TARGET"].update(
            {
                LatestInt:Answer
            }
        )
    for value in WikiSettings["CATEGORIES"]:
        LatestInt += 1
    print(LatestInt)
    for Group in AdditionalDataset_2[:Limit]:
        # for symbol in symbols:
        Question = (Group["question_text"]).replace('"',"")
        Answer = (Group["answer_text"]).replace('"',"")
        WikiDataset["dataset"].update(
            {
                Answer:{
                    "questions": [Question]
                }
            }
        )
        # LatestInt_2 = LatestInt
        LatestInt += 1
        WikiSettings["CATEGORIES"].update(
            {
                Answer: LatestInt
            }
        )

        WikiSettings["CATEGORIES_TARGET"].update(
            {
                LatestInt:Answer
            }
        )

def RuBQ():
    LatestInt = -1
    for value in WikiSettings["CATEGORIES"]:
        LatestInt += 1
    print(LatestInt)
    for Group in AdditionalDataset:
        # for symbol in symbols:
        Question = (Group["question_text"]).replace('"',"")
        Answer = (Group["answer_text"]).replace('"',"")
        WikiDataset["dataset"].update(
            {
                Answer:{
                    "questions": [Question]
                }
            }
        )
        # LatestInt_2 = LatestInt
        LatestInt += 1
        WikiSettings["CATEGORIES"].update(
            {
                Answer: LatestInt
            }
        )

        WikiSettings["CATEGORIES_TARGET"].update(
            {
                LatestInt:Answer
            }
        )

def Add_to_Settings(Name):
    LatestInt = -1
    for value in Settings["CATEGORIES"]:
        LatestInt += 1
    print(LatestInt)
    Settings["CATEGORIES"].update(
        {
            Name: LatestInt
        }
    )
    Settings["CATEGORIES_TARGET"].update(
        {
            LatestInt:Name
        }
    )

def AddCategory(Name,Value):
    if not Name in Dataset["dataset"]:
        Dataset["dataset"].update(
            {
                Name:{
                    "questions": [Value]
                }
            }
        )
        print("Success!")
        Add_to_Settings(Name)
    elif Name in Dataset["dataset"]:
        Dataset["dataset"][Name].append(Value)
        Add_to_Settings(Name)
        print("Success!")

def AddValue(NameCategory,Value):
    if NameCategory in Dataset["dataset"]:
        Dataset["dataset"][NameCategory]["questions"].append(Value)
        print("Success!")
    elif not NameCategory in Dataset["dataset"]:
        print("Category is not found in dataset.")

def AddCategory_Answers(Name,Value):
    if not Name in ArtyomSettings:
        ArtyomSettings.update(
            {
                Name: [Value]
            }
        )
        print("Success!")
    elif Name in ArtyomSettings:
        ArtyomSettings[Name].append(Value)
        print("Success!")

def AddValue_Answers(Name,Value):
    if not Name in ArtyomSettings:
        print("Category is not found in answers.")
    elif Name in ArtyomSettings:
        ArtyomSettings[Name].append(Value)
        print("Success!")

def Save():
    DatasetFile = open(os.path.join(ProjectDir,"Datasets/ArtyomDataset_2.json"),"w",encoding="utf-8")
    json.dump(Dataset,DatasetFile,ensure_ascii=False, indent=2,sort_keys=True)
    DatasetFile.close

    WikiDatasetFile = open(os.path.join(ProjectDir,"Datasets/WikiDataset.json"),"w",encoding="utf-8")
    json.dump(WikiDataset,WikiDatasetFile,ensure_ascii=False, indent=2,sort_keys=True)
    WikiDatasetFile.close

    WikiSettingsFile = open(os.path.join(ProjectDir,"NeuralNetworkSettings/WikiSettings.json"),"w",encoding="utf-8")
    json.dump(WikiSettings,WikiSettingsFile,ensure_ascii=False, indent=2,sort_keys=True)
    WikiSettingsFile.close()

    ArtyomSettingsFile = open(os.path.join(ProjectDir,"NeuralNetworkSettings/ArtyomAnswers_2.json"),"w",encoding="utf-8")
    json.dump(ArtyomSettings,ArtyomSettingsFile,ensure_ascii=False, indent=2,sort_keys=True)
    ArtyomSettingsFile.close()

    SettingsFile = open(os.path.join(ProjectDir,"NeuralNetworkSettings/Settings_2.json"),"w",encoding="utf-8")
    json.dump(Settings,SettingsFile,ensure_ascii=False, indent=2,sort_keys=True)
    SettingsFile.close()

while True:
    command = input(">>>")
    if command == "ac":
        category = input("Name Category:")
        value = input("Enter one value:")
        AddCategory(category,value)
    elif command == "mac":
        while True:
            category = input("Name Category:")
            value = input("Enter one value:")
            if value == "exit_mac":
                break
            else:
                AddCategory(category,value)
    elif command == "av":
        category = input("Name Category:")
        value = input("Value:")
        AddValue(category,value)
    elif command == "mav":
        category = input("Name Category:")
        while True:
            value = input("Value:")
            if value == 'exit_mav':
                break
            else:
                AddValue(category,value)
    elif command == 'aca':
        category = input("Name Category:")
        value = input("Enter one value:")
        AddCategory_Answers(category,value)
    elif command == 'maca':
        while True:
            category = input("Name Category:")
            value = input("Enter one value:")
            if value == 'exit_maca':
                break
            else:
                AddCategory_Answers(category,value)
    elif command == 'ava':
        category = input("Name Category:")
        value = input("Value:")
        AddValue_Answers(category,value)
    elif command == 'mava':
        category = input("Name Category:")
        while True:
            value = input("Value:")
            if value == 'exit_mava':
                break
            else:
                AddValue_Answers(category,value)
    elif command == "rubq":
        RuBQ()
    elif command == "rubq_2":
        RuBQ_2()
    elif command == "save":
        Save()
    elif command == 'exit':
        break
Save()
