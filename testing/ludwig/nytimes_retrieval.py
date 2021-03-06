import enum
from pynytimes import NYTAPI
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import datetime
import nltk
import re
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


# docs: https://pypi.org/project/pynytimes/#article-search


# functions

# just copied from: https://stackabuse.com/python-for-nlp-working-with-facebook-fasttext-library/
# observation: year numbers and citation codes remain after preprocessing
def preprocess_text(document):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(document))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    tokens = document.split()
    tokens = [nltk.stem.WordNetLemmatizer().lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if word not in set(nltk.corpus.stopwords.words('english'))]
    tokens = [word for word in tokens if len(word) > 3]

    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


# definitions
keyword = 'nature'
api_key = ''
with open('nytimes_api_key.txt', 'r') as file:
    api_key = file.read()
nyt = NYTAPI(key=api_key, parse_dates=True)
n_articles = 100

# retrieve articles for the keyword within a given time range
articles = nyt.article_search(
    query = keyword,
    results = n_articles,
    dates = {
        'begin': datetime.datetime(2020, 1, 1),
        'end': datetime.datetime(2020, 12, 31)
    },
    options = {
        'sort': 'relevance',
        'sources': 'New York Times'
    }
)

# print out all articles
list_of_articles = []
for i, article in enumerate(articles):
    headline = article['headline']['main']
    print(headline)
    list_of_articles.append(headline)

# check number of articles
print('===\nWe have', len(articles), 'articles.\n===')


# pre-process the documents
list_of_articles = preprocess_text(list_of_articles)
print('===\npre-processed text:\n', list_of_articles)


# create wordcloud and plot it
wordcloud = WordCloud().generate(list_of_articles)
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Example word cloud')
plt.axis('off')
plt.show()

"""
observations:
* word cloud currently shows only some letter
* we can also extract additional information, e.g. abstract and link to article
"""



# use a TFIDF vectorizer on the document
# (with pre-defined stuff for testing)

doc = [
    'Nature is great.',
    'Super stone.',
    'What is fire.',
    'I hate water!'
]
for i, data in enumerate(doc):
    doc[i] = preprocess_text(data)
print('===\nText', doc)

# vectorizer = TfidfVectorizer(analyzer='word')
# tfidf_matrix = vectorizer.fit_transform(doc)
# print('===\nTfIdf feature names:\n', vectorizer.get_feature_names_out())
# print('===\nThe TFIDF matrix:', tfidf_matrix.shape, '\n', tfidf_matrix)
# transformer = TfidfTransformer()
# tfidf_list = TfidfTransformer.fit_transform(tfidf_matrix)
# print('===\nTransformed list:\n', tfidf_list)

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(doc)
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)
print('===\nTfIdf-Dataframe:\n', df)
transformer = TfidfTransformer()
# tfidf_list = TfidfTransformer.fit_transform(df)
# print('===\nTransformed list:\n', tfidf_list)