import requests
from zipfile import ZipFile
import os

ProjectDir = os.path.dirname(os.path.realpath(__file__))
URL = "https://alphacephei.com/vosk/models/vosk-model-ru-0.22.zip"
response = requests.get(URL)
file = open("model_1.zip","wb")
file.write(response.content)
file.close()
# if os.path.exists(os.path.join(ProjectDir,'model')):
#     os.rmdir(os.path.join(ProjectDir,'model'))
# zf = ZipFile('model_1.zip', 'r')
# zf.extractall('model')
# zf.close()