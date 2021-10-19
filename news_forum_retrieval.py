# news forum retrieval

import datetime
from pynytimes import NYTAPI

"""
The plan:

-   define the four keywords
-   define the time periods of interest
-   retrieve the articles for each keyword and time period
-   pre-process all the documents
-   create a wordcloud for each document
-   calculate the TfIdf for all pairs for the time periods for each keyword
"""


# keywords
from datetime import datetime


keywords = [
    'nature', 'pollution', 'sustainability', 'environmentally friendly'
]

# time periods
time_periods = [
    (datetime.datetime(2000, 1, 1), datetime.datetime(2000, 12, 31)),
    (datetime.datetime(2010, 1, 1), datetime.datetime(2010, 12, 31)),
    (datetime.datetime(2020, 1, 1), datetime.datetime(2020, 12, 31)),
]

# definitions for the news forum API
api_key = ''
with open('nytimes_api_key.txt', 'r') as file:
    api_key = file.read()
nyt = NYTAPI(key=api_key, parse_dates=True)
n_articles = 100 # number of articles to retrieve with each API call

# retrieve the articles
# articles = [[], []]
for i, keyword in enumerate(keywords):
    for j, time_period in enumerate(time_periods):

    article = nyt.article_search(
        query = keyword,
        results = n_articles,
        dates = {
            'begin': time_period[0],
            'end': time_period[1]
        },
        options = {
            'sort': 'relevance',
            'sources': 'New York Times'
        }
    )

    # at this point the article has to be filtered out of all the different data and saved
    # in a proper data structure (use maybe a nested list or a dict here?)