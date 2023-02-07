import pandas as pd
import os
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

ProjectDir = os.path.dirname(os.path.realpath(__file__))

class CommunicationNetwork(BaseEstimator):
    def __init__(self,k=5,temperature=1.0):
        self.k=k
        self.temperature=temperature

    def softmax(self,t):
        out = np.exp(t)
        return out / np.sum(out, keepdims=True)

    def fit(self,TrainInput,TrainTarget):
        self.Classifier = BallTree(TrainInput)
        self.TrainTarget = np.array(TrainTarget)
    
    def predict(self,Input:str,random_state = None):
        Input = [Input]
        Input = self.vectorizer.transform(Input)
        Input = self.Compresser.transform(Input)
        distances,indices = self.Classifier.query(Input,return_distance = True,k = self.k)
        result = []
        for distance,index in zip(distances,indices):
            result.append(np.random.choice(index,p = self.softmax(distance * self.temperature)))
        return self.TrainTarget[result]

    def Start(self):
        self.Dataset = pd.read_csv(os.path.join(ProjectDir,'Datasets/CommunicationDataset.tsv'),sep = '\t')
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(self.Dataset.context_0)
        self.VectorizedData = self.vectorizer.transform(self.Dataset.context_0)
        self.Compresser = TruncatedSVD(n_components=300)
        self.Compresser.fit(self.VectorizedData)
        self.TrainInput = self.Compresser.transform(self.VectorizedData)
        self.TrainTarget = self.Dataset.reply
        self.fit(self.TrainInput,self.TrainTarget)
    
if __name__ == "__main__":
    communication_network = CommunicationNetwork()
    communication_network.Start()
    print(communication_network.predict('я не знаю никакого джейка'))