import array
from collections import Counter
from typing import OrderedDict
from numpy import single
import numpy as np
import wikipediaapi
from bs4 import BeautifulSoup
import requests
import re
import nltk
import ssl
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import pandas as pd
from itertools import combinations
from scipy import spatial
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import math
import os
import psutil
from gensim.models import KeyedVectors
from utils import Utils # import from own class


# ensure that there are no verified context errors for nltk
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')


### main ###

# create utility class instance
obj = Utils()

# Task 1: Get unprocessed pages, subsections, and list of entities (clickable keywords except reference list)
#setup()
obj.setup()

pid = os.getpid()
python_process = psutil.Process(pid)
memoryUse = python_process.memory_info().rss # memory used in bytes
#print('memory use:', memoryUse)

# Task 2: Create combined corpus, as well as separate corpuses to do TFIDF and cosine similarity
#corpus_creation(unprocessed_page, "pages")
obj.corpus_creation(obj.unprocessed_page, 'pages')
#print(all_document_corpus['nature'])
#corpus_creation(page_subsections, "subsections")
obj.corpus_creation(obj.page_subsections, 'subsections')
#print(subsections_corpus['nature'])
#corpus_creation(page_entities_list, "keywords")
obj.corpus_creation(obj.page_entities_list, 'keywords')
#print(entity_list_corpus['nature'])
#print(everything_corpus)   # combination of all documents into one corpus

all_cosine_results = array.array('d', [])
for pair in list(combinations(list(obj.single_document_corpus), 2)):
    #print(pair[0] + " " + pair[1])
    obj.tfidf_results = obj.vectorizer(obj.single_document_corpus[pair[0]],
                                   obj.single_document_corpus[pair[1]])
    #print(tfidf_results)
    all_cosine_results.append(
        obj.calculate_cosine_similarity(obj.single_document_corpus[pair[0]],
                                        obj.single_document_corpus[pair[1]]))
    obj.cosine_result = obj.calculate_cosine_similarity(
        obj.single_document_corpus[pair[0]],
        obj.single_document_corpus[pair[1]])
    #print(pair[0] + " " + pair[1])
    #print(cosine_result)
#print(all_cosine_results)

# Task 3: Repeat but with the titles of subsections
for pair in list(combinations(list(obj.subsections_corpus), 2)):
    #print(pair[0] + " " + pair[1])
    tfidf_results = obj.vectorizer(obj.subsections_corpus[pair[0]], obj.subsections_corpus[pair[1]])
    #print(tfidf_results)
    cosine_results = obj.calculate_cosine_similarity(
        obj.subsections_corpus[pair[0]], obj.subsections_corpus[pair[1]])
    #print(pair[0] + " " + pair[1])
    #print(cosine_results)

# Task 4: Repeat but with the entity-categories
for pair in list(combinations(list(obj.entity_list_corpus), 2)):
    #print(pair[0] + " " + pair[1])
    tfidf_results = obj.vectorizer(obj.entity_list_corpus[pair[0]], obj.entity_list_corpus[pair[1]])
    #print(tfidf_results)
    cosine_results = obj.calculate_cosine_similarity(
        obj.entity_list_corpus[pair[0]], obj.entity_list_corpus[pair[1]])
    #print(pair[0] + " " + pair[1])
    #print(cosine_results)

# Task 5: Calculate wu and Palmer WordNet semantic similarity, write a vector representing that similarity,
# and calculate the correlation between the semantic similarity and the wikipedia based similarity
pair_words = ['nature', 'pollution', 'sustainability', 'environment']
all_wupalmer_results = array.array('d', [])
for pair in combinations(pair_words, 2):
    all_wupalmer_results.append(obj.calculate_wupalmer(pair[0], pair[1]))
    #print(pair[0] + " " + pair[1])
    #print(calculate_wupalmer(pair[0], pair[1]))
#print(all_cosine_results)
#print(all_wupalmer_results)

#plt.scatter(all_cosine_results, all_wupalmer_results)
#plt.xlim([0.3, 1.1])
#plt.ylim([0.3, 1.1])
#plt.xlabel('cosine results')
#plt.ylabel('wu and palmer results')
#plt.title('similarity comparison')
#plt.show()

wu_wiki_correlation = pearsonr(all_wupalmer_results, all_cosine_results)
# first value is the pearson's correlation coefficient, second value is the two-tailed p-value
# negative, weak relationship (as first increases, second decreases)
#print(wu_wiki_correlation)

# Task 6: Scrape content of each entity and retrieve all clickable keywords identified
for key in obj.page_entities_list:
    one_pass_entities = ""
    for entity_category in obj.page_entities_list[key]:
        if "https://" in obj.page_entities_list[key][entity_category]:
            break
        else:
            entity_category_scrape = obj.entity_category_scraper(
                obj.page_entities_list[key][entity_category])
        one_pass_entities = one_pass_entities + " " + entity_category_scrape
        break
    obj.all_one_pass_entity_categories[key] = one_pass_entities
#print(all_one_pass_entity_categories)

# Task 7: Perform tfidf and cosine similarity on the scraped entity list
obj.corpus_creation(obj.all_one_pass_entity_categories, "keywords_2")
#print(one_pass_entity_list_corpus)
for pair in list(combinations(list(obj.one_pass_entity_list_corpus), 2)):
    #print(pair[0] + " " + pair[1])
    tfidf_results = obj.vectorizer(obj.one_pass_entity_list_corpus[pair[0]],
                               obj.one_pass_entity_list_corpus[pair[1]])
    #print(tfidf_results)
    cosine_results = obj.calculate_cosine_similarity(
        obj.one_pass_entity_list_corpus[pair[0]],
        obj.one_pass_entity_list_corpus[pair[1]])
    #print(cosine_results)

#print('memory use:', memoryUse)

# Task 8: Give the pairwise similarity scores for the keywords using a pre-trained word2vec model
# place file in local user's download folder; can be downloaded from here: https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
word2vec_model_file_path = '~/Downloads/GoogleNews-vectors-negative300.bin'
keywords = ['nature', 'pollution', 'sustainability', 'environmental']
word2vec_scores = obj.calc_word2vec_scores(keywords, word2vec_model_file_path)

# Task 9: Retrieve news forum data, process it and show word cloud presentations
n_news_forum_articles = 2 
obj.retrieve_articles(n_news_forum_articles, obj.news_forum_data)
obj.display_word_cloud_represenations(obj.news_forum_data)

# Task 10: Calculate the Tf-Idf scores for each keyword pair for each time period based on the news forum data
print('check data first:\n')
print(obj.news_forum_data['nature']['2001-2004']['doc'])
print(obj.news_forum_data['pollution']['2001-2004']['doc'])
print(obj.news_forum_data)
obj.calc_tfidf_scores_news_forum(obj.news_forum_data)