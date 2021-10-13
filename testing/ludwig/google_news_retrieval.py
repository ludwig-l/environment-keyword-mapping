# testing with retrieval from google news and stuff ... (point number 9 on the list)

# docs: https://pypi.org/project/GoogleNews/

from GoogleNews import GoogleNews
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# set up using a given time range
start_date = '01/10/2021' # watch out for mm/dd/yyyy format
end_date = '07/10/2021' # watch out for mm/dd/yyyy format
googlenews = GoogleNews(start=start_date, end=end_date)

# retrieve some news
googlenews.get_news('nature')
news = googlenews.get_texts() # extract all titles of the search results
print(news, '\nLength:', len(news))


"""
observations:
-   the search() function only outputs the the title of the news and some other stuff but not
    the content of the news
-   but the titles can be extracted quite easily at least
-   the project description only says "retrieve documents for each keyword", so maybe that is
    our choice then (if titles only or full text of the articles)
"""


# now let's try to put the news results into a data construct

# define the time periods of interest
time_periods = [('01/01/2010', '12/31/2010'),
                ('01/01/2015', '12/31/2015'),
                ('01/01/2020', '12/31/2020'),
                ('01/01/2021', '01/31/2021')]

gnews = GoogleNews()
data = [[] for _ in range(len(time_periods))] # create a list of n empty lists
print('========\ndataframe:', data, '=========')
for i, time_period in enumerate(time_periods):
    # assign the list of news titles to the list of lists
    gnews.clear()
    gnews.set_time_range(time_period[0], time_period[1])
    gnews.get_news('nature')
    news = gnews.get_texts()
    data[i] = news

print('The data looks like:\n', data)

print('===\nThe individual number of documents is:')
for i, time_period in enumerate(time_periods):
    print('-> Time period', time_periods[i], 'has length', len(data[i]))


# seems like a just a list of lists is the easiest here (possible unequal amounts of documents per
# time period is the challange here)

# also it seems like the library only gives a maximum of 92 news articles regardless of the length
# of the time periods (we need a few hundred documents for each time period if I understood it
# correctly)


# word cloud representation
for i, time_period in enumerate(time_periods):
    whole_str = ' '.join(data[i]) # join all the strings in the list to one string
    wordcloud = WordCloud().generate(whole_str)
    # plot
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(time_period)
    plt.axis('off')
    plt.show()