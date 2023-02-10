import requests
from bs4 import BeautifulSoup
from loguru import logger
import json
import os

class ParseNews:
    def __init__(self,ProjectDir) -> None:
        self.NewsURL = "https://lenta.ru/"
        self.itNewsURL = ""
        self.NewsArray = []
        self.itNewsArray = []
        self.ProjectDir = ProjectDir

    def News(self):
        r = requests.get(self.NewsURL).text
        soup = BeautifulSoup(r, 'html.parser')
        content = soup.find_all("span")
        for span in content:
            text = span.text.replace('"','')
            # text = await self.FilteringTransforms(text,to_words=True)
            self.NewsArray.append(text)
        file = open(self.ProjectDir + "/AssistantSettings/News.json","w")
        json.dump(self.NewsArray,file,indent=2,ensure_ascii=False,sort_keys=True)
        file.close()

    def it_news(self):
        pass

    def StartParse(self):
        self.News()
        self.it_news()

if __name__ == "__main__":
    Parse = ParseNews(os.path.dirname(os.path.realpath(__file__)))
    Parse.StartParse()