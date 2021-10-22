# news forum retrieval

import datetime
from pynytimes import NYTAPI
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer

"""
The plan:

-   define the four keywords
-   define the time periods of interest
-   retrieve the articles for each keyword and time period
-   pre-process all the documents
-   create a wordcloud for each document
-   calculate the TfIdf for all pairs for the time periods for each keyword
"""


# data structure for storing all
data  = {
    'nature' : {
        '2000' : {
            'titles' : [],
            'urls' : []
        },
        '2005' : {
            'titles' : [],
            'urls' : []
        },
        '2010' : {
            'titles' : [],
            'urls' : []
        },
        '2015' : {
            'titles' : [],
            'urls' : []
        },
        '2020' : {
            'titles' : [],
            'urls' : []
        },
    },
    'pollution' : {
        '2000' : {
            'titles' : [],
            'urls' : []
        },
        '2005' : {
            'titles' : [],
            'urls' : []
        },
        '2010' : {
            'titles' : [],
            'urls' : []
        },
        '2015' : {
            'titles' : [],
            'urls' : []
        },
        '2020' : {
            'titles' : [],
            'urls' : []
        },
    },
    'sustainability' : {
        '2000' : {
            'titles' : [],
            'urls' : []
        },
        '2005' : {
            'titles' : [],
            'urls' : []
        },
        '2010' : {
            'titles' : [],
            'urls' : []
        },
        '2015' : {
            'titles' : [],
            'urls' : []
        },
        '2020' : {
            'titles' : [],
            'urls' : []
        },
    },
    'environmentally friendly' : {
        '2000' : {
            'titles' : [],
            'urls' : []
        },
        '2005' : {
            'titles' : [],
            'urls' : []
        },
        '2010' : {
            'titles' : [],
            'urls' : []
        },
        '2015' : {
            'titles' : [],
            'urls' : []
        },
        '2020' : {
            'titles' : [],
            'urls' : []
        },
    }
}


# definitions for the news forum API
api_key = ''
with open('nytimes_api_key.txt', 'r') as file:
    api_key = file.read()
nyt = NYTAPI(key=api_key, parse_dates=True)
n_articles = 10 # number of articles to retrieve with each API call


# retrieve the articles
for keyword in data:
    print(keyword)
    for year in data[keyword]:

        print('---', year) # progress information

        # define the two needed datetime objects
        date_begin = datetime.datetime.fromisoformat(year + '-01-01')
        date_end = datetime.datetime.fromisoformat(year + '-12-31')

        # API call for the desired information
        retrieved_articles = nyt.article_search(
            query = keyword,
            results = n_articles,
            dates = {
                'begin': date_begin,
                'end': date_end
            },
            options = {
                'sort': 'relevance',
                'sources': 'New York Times'
            }
        )

        # collect data of interest
        for i, article_data in enumerate(retrieved_articles):
            data[keyword][year]['titles'].append(article_data['headline']['main'])
            data[keyword][year]['urls'].append(article_data['web_url'])


# print all the stuff
print('===\nHere is the stuff you wanted:\n', data)


# here do some kind of pre-processing


# here will be the word cloud representation


# test the TfIdf functionality on one data set
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([
    # use only the 'nature' keyword here
    ' '.join(data['nature']['2000']['titles']),
    ' '.join(data['nature']['2010']['titles']),
    ' '.join(data['nature']['2020']['titles'])
])
feature_names = vectorizer.get_feature_names_out()
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)
print('===\nTfIdf-Dataframe:\n', df)