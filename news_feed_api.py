import requests
import json
import csv

url = ('https://newsapi.org/v2/everything?sources=the-times-of-india&q=hyderabad&from=2018-03-10&sortBy=popularity&apiKey=56638d1f5c28446393e3f050488236fd')
response = requests.get(url)
print (response.json())

news_data =  response.json()

news_parsed = json.loads(news_data)
article_data = news_parsed['articles']
article_csv = open('everything.csv', 'w')
csvwriter = csv.writer(article_csv)
# use encode to convert non-ASCII characters
count = 0
for article in article_data:
    if count == 0:
        header = article.keys()
        csvwriter.writerow(header)
        count += 1
    csvwriter.writerow(article.values())
article_csv.close()
