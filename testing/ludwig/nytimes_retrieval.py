from pynytimes import NYTAPI
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import datetime


# docs: https://pypi.org/project/pynytimes/#article-search


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

# create wordcloud and plot it
whole_str = ' '.join(list_of_articles) # join all the strings in the list to one string
wordcloud = WordCloud().generate(whole_str)
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Example word cloud')
plt.axis('off')
plt.show()

"""
observations:
* word cloud currently shows only some letter
* we can also extract additional information, e.g. abstract and link to article
"""