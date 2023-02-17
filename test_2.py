# 1
# import pandas as pd
# import os
# import json
# ProjectDir = os.path.dirname(os.path.realpath(__file__))
# DatasetFile = open(os.path.join(ProjectDir,'Datasets/ArtyomDataset3.json'),'r',encoding="utf-8")
# Dataset = json.load(DatasetFile)
# DatasetFile.close()
# CommunicationDataset = pd.read_csv(os.path.join(ProjectDir,'Datasets/CommunicationDataset.tsv'),sep = '\t')
# for value in CommunicationDataset.reply[:4096]:
#     Dataset["dataset"]["communication"]["questions"].append(value)
#     print(value)
# DatasetFile = open(os.path.join(ProjectDir,"Datasets/ArtyomDataset3.json"),"w",encoding="utf-8")
# json.dump(Dataset,DatasetFile,ensure_ascii=False, indent=2,sort_keys=True)
# DatasetFile.close
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
from plyer import notification  
  
notification_title = 'GREETINGS FROM JAVATPOINT!'  
notification_message = 'Thank you for reading. Have a Good Day.'  
  
notification.notify(  
    title = notification_title,  
    message = notification_message,  
    app_icon = None,  
    timeout = 10,  
    toast = False  
    )  