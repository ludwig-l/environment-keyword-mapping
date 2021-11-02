# news forum retrieval

import datetime
from pynytimes import NYTAPI
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from itertools import combinations
from wordcloud import WordCloud
import matplotlib.pyplot as plt

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


# taken from other main script
def cosine_similarity(document_1, document_2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([document_1, document_2])
    return ((vectors * vectors.T).A)[0,1]


# data structure for storing all
data  = {
    'nature' : {
        '2001-2004' : {
            'titles' : [],
            'urls' : [],
            'abstracts' : [],
            'snippets' : [],
            'lead_paragraphs' : [],
            'doc' : ''
        },
        '2005-2008' : {
            'titles' : [],
            'urls' : [],
            'abstracts' : [],
            'snippets' : [],
            'lead_paragraphs' : [],
            'doc' : ''
        },
        '2009-2012' : {
            'titles' : [],
            'urls' : [],
            'abstracts' : [],
            'snippets' : [],
            'lead_paragraphs' : [],
            'doc' : ''
        },
        '2013-2016' : {
            'titles' : [],
            'urls' : [],
            'abstracts' : [],
            'snippets' : [],
            'lead_paragraphs' : [],
            'doc' : ''
        },
        '2017-2020' : {
            'titles' : [],
            'urls' : [],
            'abstracts' : [],
            'snippets' : [],
            'lead_paragraphs' : [],
            'doc' : ''
        },
    },
    'pollution' : {
        '2001-2004' : {
            'titles' : [],
            'urls' : [],
            'abstracts' : [],
            'snippets' : [],
            'lead_paragraphs' : [],
            'doc' : ''
        },
        '2005-2008' : {
            'titles' : [],
            'urls' : [],
            'abstracts' : [],
            'snippets' : [],
            'lead_paragraphs' : [],
            'doc' : ''
        },
        '2009-2012' : {
            'titles' : [],
            'urls' : [],
            'abstracts' : [],
            'snippets' : [],
            'lead_paragraphs' : [],
            'doc' : ''
        },
        '2013-2016' : {
            'titles' : [],
            'urls' : [],
            'abstracts' : [],
            'snippets' : [],
            'lead_paragraphs' : [],
            'doc' : ''
        },
        '2017-2020' : {
            'titles' : [],
            'urls' : [],
            'abstracts' : [],
            'snippets' : [],
            'lead_paragraphs' : [],
            'doc' : ''
        },
    },
    'sustainability' : {
        '2001-2004' : {
            'titles' : [],
            'urls' : [],
            'abstracts' : [],
            'snippets' : [],
            'lead_paragraphs' : [],
            'doc' : ''
        },
        '2005-2008' : {
            'titles' : [],
            'urls' : [],
            'abstracts' : [],
            'snippets' : [],
            'lead_paragraphs' : [],
            'doc' : ''
        },
        '2009-2012' : {
            'titles' : [],
            'urls' : [],
            'abstracts' : [],
            'snippets' : [],
            'lead_paragraphs' : [],
            'doc' : ''
        },
        '2013-2016' : {
            'titles' : [],
            'urls' : [],
            'abstracts' : [],
            'snippets' : [],
            'lead_paragraphs' : [],
            'doc' : ''
        },
        '2017-2020' : {
            'titles' : [],
            'urls' : [],
            'abstracts' : [],
            'snippets' : [],
            'lead_paragraphs' : [],
            'doc' : ''
        },
    },
    'environmentally friendly' : {
        '2001-2004' : {
            'titles' : [],
            'urls' : [],
            'abstracts' : [],
            'snippets' : [],
            'lead_paragraphs' : [],
            'doc' : ''
        },
        '2005-2008' : {
            'titles' : [],
            'urls' : [],
            'abstracts' : [],
            'snippets' : [],
            'lead_paragraphs' : [],
            'doc' : ''
        },
        '2009-2012' : {
            'titles' : [],
            'urls' : [],
            'abstracts' : [],
            'snippets' : [],
            'lead_paragraphs' : [],
            'doc' : ''
        },
        '2013-2016' : {
            'titles' : [],
            'urls' : [],
            'abstracts' : [],
            'snippets' : [],
            'lead_paragraphs' : [],
            'doc' : ''
        },
        '2017-2020' : {
            'titles' : [],
            'urls' : [],
            'abstracts' : [],
            'snippets' : [],
            'lead_paragraphs' : [],
            'doc' : ''
        },
    }
}


# definitions for the news forum API
api_key = ''
with open('nytimes_api_key.txt', 'r') as file:
    api_key = file.read()
nyt = NYTAPI(key=api_key, parse_dates=True)
n_articles = 250 # number of articles to retrieve with each API call


# retrieve the articles
for keyword in data:
    print(keyword)
    for year in data[keyword]:

        print('---', year) # progress information

        # define the two needed datetime objects
        date_begin = datetime.datetime.fromisoformat(year[:4] + '-01-01')
        date_end = datetime.datetime.fromisoformat(year[5:] + '-12-31')

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
            data[keyword][year]['abstracts'].append(article_data['abstract'])
            data[keyword][year]['snippets'].append(article_data['snippet'])
            data[keyword][year]['lead_paragraphs'].append(article_data['lead_paragraph'])

        # join each title together to one document and pre-process the text
        #data[keyword][year]['doc'] = preprocess_and_lemmatize(' '.join(data[keyword][year]['titles']))
        data[keyword][year]['doc'] = preprocess_and_lemmatize(' '.join(data[keyword][year]['abstracts']))


# now let't implement this score computation for all the possible pairs
print('===\nCosine similarity scores:')
for pair in list(combinations(data, 2)):
    for year in data['nature']: # just using the first entry here for simplicity (ad)
        score = cosine_similarity(data[pair[0]][year]['doc'], data[pair[1]][year]['doc'])
        print('-> Score for', pair, 'for years', year, score)


# create word clouds and plot them
for keyword in data:
    for year in data[keyword]:
        # generate the wordcloud from the documents
        wordcloud = WordCloud().generate(data[keyword][year]['doc'])
        # plot
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig('word_cloud_' + keyword + '_' + year + '.png', bbox_inches = 'tight', pad_inches = 0)
        # save the figure before adding title and after turing off the axis, otherwise the saved picture will be ugly
        plt.title('Word cloud representation for keyword \"' + keyword + '\" for time period ' + year)
        plt.show()