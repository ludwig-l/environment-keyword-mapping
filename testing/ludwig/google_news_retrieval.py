# testing with retrieval from google news and stuff ...

# docs: https://pypi.org/project/GoogleNews/

from GoogleNews import GoogleNews

# set up using a given time range
start_date = '01/10/2021' # watch out for mm/dd/yyyy format
end_date = '07/10/2021' # watch out for mm/dd/yyyy format
googlenews = GoogleNews(start=start_date, end=end_date)

# retrieve some news
googlenews.get_news('nature')
news = googlenews.get_texts() # extract all titles of the search results
print(news)

# select one specific page
# print(googlenews.results()[0])


"""
observations:
-   the search() function only outputs the the title of the news and some other stuff but not
    the content of the news
-   but the titles can be extracted quite easily at least
-   the project description only says "retrieve documents for each keyword", so maybe that is
    our choice then (if titles only or full text of the articles)
"""