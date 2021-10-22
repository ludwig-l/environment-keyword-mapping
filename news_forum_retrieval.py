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


# functions

# this function is currently just copied from other main script

def preprocess_and_lemmatize(document):      
    corpus_part = ""
    # preprocess
    # to lowercase
    document = document.lower()
    # remove symbols/special characters
    document = re.sub(r'\W', ' ', str(document))
    # remove single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    # remove single characters from the first characters
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
    # standardize number of spaces >1 space becomes 1 space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    # remove any prefixed "b"
    document = re.sub(r'^b\s+', '', document)
    # remove numbers that are not 20th or 21st century years
    document = re.sub(r'\b(?!(\D\S*|[12][0-9]{3})\b)\S+\b', '', document)
    # lemmatize
    stemmer = WordNetLemmatizer()
    english_stop = set(nltk.corpus.stopwords.words('english'))
    tokens = document.split()
    tokens = [stemmer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if word not in english_stop]
    # keep words that are greater than 2 characters
    tokens = [word for word in tokens if len(word) > 2]
    processed_document = ' '.join(tokens)
    corpus_part = corpus_part + processed_document
    return corpus_part


# data structure for storing all
data  = {
    'nature' : {
        '2000' : {
            'titles' : [],
            'urls' : [],
            'doc' : ''
        },
        '2005' : {
            'titles' : [],
            'urls' : [],
            'doc' : ''
        },
        '2010' : {
            'titles' : [],
            'urls' : [],
            'doc' : ''
        },
        '2015' : {
            'titles' : [],
            'urls' : [],
            'doc' : ''
        },
        '2020' : {
            'titles' : [],
            'urls' : [],
            'doc' : ''
        },
    },
    'pollution' : {
        '2000' : {
            'titles' : [],
            'urls' : [],
            'doc' : ''
        },
        '2005' : {
            'titles' : [],
            'urls' : [],
            'doc' : ''
        },
        '2010' : {
            'titles' : [],
            'urls' : [],
            'doc' : ''
        },
        '2015' : {
            'titles' : [],
            'urls' : [],
            'doc' : ''
        },
        '2020' : {
            'titles' : [],
            'urls' : [],
            'doc' : ''
        },
    },
    'sustainability' : {
        '2000' : {
            'titles' : [],
            'urls' : [],
            'doc' : ''
        },
        '2005' : {
            'titles' : [],
            'urls' : [],
            'doc' : ''
        },
        '2010' : {
            'titles' : [],
            'urls' : [],
            'doc' : ''
        },
        '2015' : {
            'titles' : [],
            'urls' : [],
            'doc' : ''
        },
        '2020' : {
            'titles' : [],
            'urls' : [],
            'doc' : ''
        },
    },
    'environmentally friendly' : {
        '2000' : {
            'titles' : [],
            'urls' : [],
            'doc' : ''
        },
        '2005' : {
            'titles' : [],
            'urls' : [],
            'doc' : ''
        },
        '2010' : {
            'titles' : [],
            'urls' : [],
            'doc' : ''
        },
        '2015' : {
            'titles' : [],
            'urls' : [],
            'doc' : ''
        },
        '2020' : {
            'titles' : [],
            'urls' : [],
            'doc' : ''
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