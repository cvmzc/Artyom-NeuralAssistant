import requests
from bs4 import BeautifulSoup
from loguru import logger
import json
import os
import time
import sys

class ParseNews:
    def __init__(self,ProjectDir) -> None:
        self.NewsURL = "https://lenta.ru/parts/news"
        self.itNewsURL = "https://habr.com/ru/news/"
        self.NewsArray = []
        self.itNewsArray = {}
        self.ProjectDir = ProjectDir
        sys.path.append(self.ProjectDir)
        sys.path.append(os.path.join(self.ProjectDir,"Transforms"))
        from Transforms.FilteringTransforms import FilteringTransforms
        self.Transform = FilteringTransforms()

    def News(self):
        r = requests.get(self.NewsURL).text
        soup = BeautifulSoup(r, 'html.parser')
        content = soup.find_all("h3",class_ = 'card-full-news__title')
        for span in content:
            text = span.text.replace('"','')
            # text = await self.FilteringTransforms(text,to_words=True)
            self.NewsArray.append(text)
        file = open(os.path.join(self.ProjectDir,"AssistantSettings/News.json"),"w",encoding="utf-8")
        json.dump(self.NewsArray,file,indent=2,ensure_ascii=False,sort_keys=True)
        file.close()

    def it_news(self):
        try:
            r = requests.get(self.itNewsURL,headers={"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"}).text
            soup = BeautifulSoup(r, 'html.parser')
            time.sleep(1)
            content = soup.find_all("a",class_='tm-article-snippet__title-link',href=True)
            time.sleep(1)
            for span in content:
                title = span.text.replace('"','')
                title = self.Transform.group_normalize(title)
                time.sleep(1)
                link = span.get("href")
                time.sleep(1)
                print(link)
                r_page = requests.get("https://habr.com" + link,headers={"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"}).text
                page = BeautifulSoup(r_page, 'html.parser')
                time.sleep(1)
                content_page = page.find('div', class_='tm-article-body')
                time.sleep(1)
                for x in content_page.find_all('p'):
                    time.sleep(1)
                    text = x.text
                    if text != None and text != '' and not title in self.itNewsArray:
                        text = self.Transform.group_normalize(text)
                        self.itNewsArray.update({title:[text]})
                    else:
                        continue
                    time.sleep(1)
                time.sleep(1)
        except:
            file = open(os.path.join(self.ProjectDir,"AssistantConfig/IT_News.json"),"w",encoding="utf-8")
            json.dump(self.itNewsArray,file,ensure_ascii=False,sort_keys=True, indent=2)
            file.close()
        print(self.itNewsArray)
            # for s in page.find('div', class_='tm-article-body').find_all('p'):
            #     print(s.text)
            # print(title)

            # text = await self.FilteringTransforms(text,to_words=True)
        #     self.itNewsArray.append(text)
        

    def StartParse(self):
        self.News()
        self.it_news()

if __name__ == "__main__":
    Parse = ParseNews(r"C:\Users\Blackflame576\Documents\Blackflame576\DigitalBit\Artyom-NeuralAssistant")
    Parse.it_news()
    # Parse.StartParse()