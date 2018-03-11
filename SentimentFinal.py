import argparse
import csv
import numpy as np

from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

def analyze(movie_review_filename):
    """Run a sentiment analysis request on text within a passed filename."""
    client = language.LanguageServiceClient()

    with open(movie_review_filename, 'r') as review_file:
        # Instantiates a plain text document.
        content = review_file.read()

    document = types.Document(
        content=content,
        type=enums.Document.Type.PLAIN_TEXT)
    annotations = client.analyze_sentiment(document=document)
    score = annotations.document_sentiment.score
    magnitude = annotations.document_sentiment.magnitude
    L=len(annotations.sentences)
#    return [score, magnitude/L]
    return [score]

    # Print the results
#    print_result(annotations)

if __name__ == '__main__':
#    parser = argparse.ArgumentParser(
#        description=__doc__,
#        formatter_class=argparse.RawDescriptionHelpFormatter)
#    parser.add_argument(
#        'movie_review_filename',
#        help='The filename of the movie review you\'d like to analyze.')
#    args = parser.parse_args()

#    arrays=["articles/test.txt","articles/test3.txt","articles/test2.txt"]
#    for X in arrays:
#        analyze(X)
    articles_path = "Articles"
    X=np.empty([0,2])

    count=0
    lastval=-10000
    lastresult=0
    flag=0
    mydict={}
    with open('EventArticles.csv') as csvfile:
        readCSV=csv.reader(csvfile,delimiter=',')
        for row in readCSV:
            sentiment_result=analyze(articles_path+"/"+row[1])
            #vprint row[1]
            #vprint sentiment_result[0]
            if flag<>0 and lastval<>row[0]:
                lastresult=lastresult/count
                count=0
                mydict[lastval]=lastresult
                row_new=[lastval,lastresult]
                X=np.vstack([X,row_new])
                #vprint row_new
                lastresult=0
            flag=1
            lastval=row[0]
            count=count+1
            lastresult=lastresult+sentiment_result[0]
        
        lastresult=lastresult/count
        count=0
        mydict[lastval]=lastresult
        row_new=[lastval,lastresult]
        X=np.vstack([X,row_new])
        #vprint row_new
        lastresult=0
    
#    result=analyze("test.txt")
#    with open('test_sentiment.csv','wb') as csvWriteFile:
#        writeCSV=csv.writer(csvWriteFile,delimiter=',')
#        print result 
    with open('Scores.csv','wb') as csvWriteFile:
        writeCSV=csv.writer(csvWriteFile,delimiter=",")
        writeCSV.writerows(X)
    
    crimepattern=np.empty([0,2])
    with open('Crimelist.csv') as csvfile:
        readCSV=csv.reader(csvfile,delimiter=',')
        for row in readCSV:
            score = mydict[row[0]]
            crimerate = float(row[1])*100
            row_new=[score,crimerate]
            crimepattern=np.vstack([crimepattern,row_new])
    print (crimepattern)

    with open('crimepattern.csv','wb') as csvWriteFile:
        writeCSV=csv.writer(csvWriteFile,delimiter=",")
        writeCSV.writerows(crimepattern)
      
