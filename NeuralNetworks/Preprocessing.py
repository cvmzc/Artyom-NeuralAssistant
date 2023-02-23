# Импортирование необходмых библиотек в целях подготовки датасета для нейросети
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import json
import random
import librosa
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from rich.progress import track
import pandas

# Подготовка датасета
ProjectDir = os.path.dirname(os.path.realpath(__file__))

class PreprocessingDataset:
    def __init__(self,BaseCategoryPredict:bool = None):
        global InputDataset
        if BaseCategoryPredict == True:
            InputDatasetFile = open(os.path.join(ProjectDir,'Datasets/InputDataset.json'), "r", encoding ='utf8')
            InputDataset = json.load(InputDatasetFile)
            InputDatasetFile.close()
        self.Dictionary = {}
        self.TrainInput = []
        self.TrainTarget = []
        self.TestInput = []
        self.TestTarget = []
        self.PredictInput = []
        self.PredictArray = []
        self.Mode = 'train'
        self.x = []
        self.y = []

    def ToMatrix(self,array):
        return np.squeeze(array)

    def ToNumpyArray(self,array):
        return np.array(array)

    def ASR_Preprocessing(self,PathAudio:str,Mode:str = 'train'):
        pass

    def TTS_Preprocessing(self,PathAudio:str,Mode:str = 'train'):
        self.PathAudio = PathAudio
        self.Mode = Mode
        if self.Mode == "train":
            self.DatasetFiles = list(os.walk(self.PathAudio))
            for (root,dirs,files) in track(os.walk(self.PathAudio,topdown=True),description='[green]Preprocessing'):
                for file in files:
                    if file.endswith('.wav'):
                        self.AudioFile = os.path.join(root,file)
                        audio,sample_rate = librosa.load(self.AudioFile,mono=True)
                        # print(audio)
                        mfcc = librosa.feature.mfcc(y = audio,n_fft=512,n_mfcc=13,n_mels=40,hop_length=160,fmin=0,fmax=None,htk=False, sr = sample_rate)
                        mfcc = np.mean(mfcc.T,axis=0)
                        self.x.append(np.array(mfcc))
                    elif file.endswith('.txt'):
                        file = open(os.path.join(root,file),'r+',encoding="utf-8")
                        DataFile = file.read()
                        self.y.append(DataFile)
                        file.close()
            InputDatasetFile = open("Datasets/TTSInputDataset.json", "w", encoding ='utf-8')
            json.dump(self.y, InputDatasetFile,ensure_ascii=False,sort_keys=True, indent=2)
            InputDatasetFile.close()
        elif self.Mode == "predict":
            pass

    def PreprocessingAudio(self,PathAudio:str,mode:str = 'train',TypeSpeech:str = 'TTS'):
        if TypeSpeech == "TTS":
            self.TTS_Preprocessing(PathAudio = PathAudio,Mode = mode)
        elif TypeSpeech == "ASR":
            self.ASR_Preprocessing(PathAudio = PathAudio,Mode = mode)
        if self.Mode == 'train' or self.Mode == 'test':
            self.DatasetFiles = list(os.walk(self.PathAudio))
            for (root,dirs,files) in track(os.walk(self.PathAudio,topdown=True),description='[green]Preprocessing'):
                for file in files[:2]:
                    if file.endswith('.wav'):
                        self.AudioFile = os.path.join(root,file)
                        audio,sample_rate = librosa.load(self.AudioFile,mono=True)
                        # print(audio)
                        mfcc = librosa.feature.mfcc(y = audio,n_fft=512,n_mfcc=13,n_mels=40,hop_length=160,fmin=0,fmax=None,htk=False, sr = sample_rate)
                        mfcc = np.mean(mfcc.T,axis=0)
                        self.x.append(np.array(mfcc))
                    elif file.endswith('.txt'):
                        file = open(os.path.join(root,file),'r+',encoding="utf-8")
                        DataFile = file.read()
                        self.y.append(DataFile)
                        file.close()
            InputDatasetFile = open("Datasets/TTSInputDataset.json", "w", encoding ='utf-8')
            json.dump(self.y, InputDatasetFile,ensure_ascii=False,sort_keys=True, indent=2)
            InputDatasetFile.close()
            
        elif self.Mode == 'predict':
            # InputDatasetFile = open("Datasets/SpeechInputDataset.json", "r", encoding ='utf8')
            # DataFile = json.load(InputDatasetFile)
            # InputDatasetFile.close()
            
            self.AudioFile = self.PathAudio
            audio,sample_rate = librosa.load(self.AudioFile,res_type='kaiser_fast')
            mfccs = librosa.feature.mfcc(y=audio,sr=sample_rate)
            mfccs = np.mean(mfccs.T,axis=0)
            self.PredictInput = self.ToMatrix(mfccs)
            return self.PredictInput

    def PreprocessingText(self,PredictArray:list = [],Dictionary:dict = {},mode = 'train'):
        global InputDataset
        if os.path.exists(os.path.join(ProjectDir,'Settings/Settings.json')):
            file = open(os.path.join(ProjectDir,'Settings/Settings.json'),'r',encoding='utf-8')
            DataFile = json.load(file)
            CATEGORIES = DataFile['CATEGORIES']
            file.close()
        else:
            raise FileNotFoundError
        
        self.Mode = mode
        if self.Mode == 'train' or self.Mode == 'test':
            self.Dictionary = list(Dictionary.items())
            random.shuffle(self.Dictionary)
            self.Dictionary = dict(self.Dictionary)
            for intent in track(self.Dictionary,description='[green]Preprocessing dataset'):
                for questions in Dictionary[intent]['questions']:
                    self.x.append(questions)
                    self.y.append(intent)
            if self.Mode == 'train':
                for target in self.y:
                    self.TrainTarget.append(CATEGORIES[target])
            elif self.Mode == 'test':
                for target in self.y:
                    self.TestTarget.append(CATEGORIES[target])
            vectorizer = TfidfVectorizer()
            vectorizer = vectorizer.fit_transform(self.x)
            VectorizedData = vectorizer.toarray()
            InputDatasetFile = open(os.path.join(ProjectDir,"Datasets/InputDataset.json"), "w", encoding ='utf8')
            json.dump(self.x, InputDatasetFile,ensure_ascii=False,sort_keys=True, indent=2)
            InputDatasetFile.close()
            if self.Mode == 'train':
                self.TrainInput = self.ToMatrix(VectorizedData)
                return self.TrainInput,self.TrainTarget
            elif self.Mode == 'test':
                self.TestInput = self.ToMatrix(VectorizedData)
                return self.TestInput,self.TestTarget

        elif self.Mode == 'predict':
            self.PredictArray = PredictArray
            vectorizer = TfidfVectorizer()
            if InputDataset:
                vectorizer.fit_transform(InputDataset)
            else:
                InputDatasetFile = open("Datasets/InputDataset.json", "r", encoding ='utf8')
                InputDataset = json.load(InputDatasetFile)
                InputDatasetFile.close()
                vectorizer.fit_transform(InputDataset)
            self.PredictInput = self.ToMatrix(vectorizer.transform(self.PredictArray).toarray())
            return self.PredictInput
