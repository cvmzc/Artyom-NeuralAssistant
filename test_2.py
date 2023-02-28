# 1
import pandas as pd
import os
import json
import re
ProjectDir = os.path.dirname(os.path.realpath(__file__))
DatasetFile = open(os.path.join(ProjectDir,'NeuralNetworks/Datasets/ArtyomDataset.json'),'r',encoding="utf-8")
Dataset = json.load(DatasetFile)
DatasetFile.close()
CommunicationDataset = pd.read_csv(os.path.join(ProjectDir,'NeuralNetworks/Datasets/CommunicationDataset.tsv'),sep = '\t')
for value in CommunicationDataset.reply[:128]:
    value = value.strip()
    value = value.lower()
    value = re.sub(r'[^\w\s]','', value)
    Dataset["dataset"]["communication"]["questions"].append(value)
    print(value)
DatasetFile = open(os.path.join(ProjectDir,"NeuralNetworks/Datasets/ArtyomDataset.json"),"w",encoding="utf-8")
json.dump(Dataset,DatasetFile,ensure_ascii=False, indent=2,sort_keys=True)
DatasetFile.close()
# 2
# from pyowm import OWM
# import geocoder
# owm = OWM('2221d769ed67828e858caaa3803161ea')
# geolocation = geocoder.ip('me')
# coordinates = geolocation.latlng
# # location = str(geolocation.address.split(',')[4]).lower()

# mgr = owm.geocoding_manager()
# city = mgr.reverse_geocode(lat=coordinates[0], lon=coordinates[1])

# mgr = owm.weather_manager()
# observation = mgr.weather_at_place(city[0].name)  # the observation object is a box containing a weather object
# weather = observation.weather
# temp = int(weather.temperature('celsius')["temp"])
# wind = weather.wind()["speed"]
# wind = f"{wind} метров в секунду"
# humidity = str(weather.humidity) + " процентов"
# 3
# import json
# import os
# import re
# ProjectDir = os.path.dirname(os.path.realpath(__file__))
# DatasetFile = open('NeuralNetworks/Datasets/ArtyomDataset.json','r',encoding="utf-8")
# Dataset = json.load(DatasetFile)
# DatasetFile.close()
# AdditionalDatasetFile = open(os.path.join(ProjectDir,"NeuralNetworks/Datasets/RuBQ_2.0.json"),"r",encoding="utf-8")
# AdditionalDataset = json.load(AdditionalDatasetFile)
# AdditionalDatasetFile.close()

# AdditionalDatasetFile_2 = open(os.path.join(ProjectDir,"NeuralNetworks/Datasets/RuBQ_2.0_test.json"),"r",encoding="utf-8")
# AdditionalDataset_2 = json.load(AdditionalDatasetFile_2)
# AdditionalDatasetFile_2.close()
# for Group in AdditionalDataset[:64]:
#     # for symbol in symbols:
#     Question = (Group["question_text"]).replace('"',"")
#     Question = Question.lower()
#     Question = re.sub(r'[^\w\s]','', Question)
#     Dataset["dataset"]["wikipedia"]["questions"].append(Question)
#     print(Question)
# for Group in AdditionalDataset_2[:64]:
#     Question_2 = (Group["question_text"]).replace('"',"")
#     Question_2 = Question_2.lower()
#     Question_2 = re.sub(r'[^\w\s]','', Question_2)
#     Dataset["dataset"]["wikipedia"]["questions"].append(Question_2)
#     print(Question_2)
# print(Dataset)
# DatasetFile = open("NeuralNetworks/Datasets/ArtyomDataset.json","w",encoding="utf-8")
# json.dump(Dataset,DatasetFile,ensure_ascii=False, indent=2,sort_keys=True)
# DatasetFile.close()
# 4
# test_1 = range(500)
# test_2 = range(500)
# def batch(iter):
#     temp_array_1 = []
#     for batch_index,value_1 in enumerate(iter):
#         if batch_index % 50 == 0:
#             yield temp_array_1
#             temp_array_1 = []
#         else:
#             temp_array_1.append(value_1)
# for value,value_2 in zip(batch(test_1),batch(test_2)):
#     input_t = value
#     print(input_t)
# def batch(iterable, n = 1):
#    current_batch = []
#    for item in iterable:
#        current_batch.append(item)
#        if len(current_batch) == n:
#            yield current_batch
#            current_batch = []
#    if current_batch:
#        yield current_batch

# for value_1,value_2 in zip(batch(test_1,50),batch(test_2)):
#     print(value_1)