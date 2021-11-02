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
from utils import Utils # import from own class


## global variables to store results
#global everything_corpus
#global all_document_corpus
#global all_document_tfidf_results
#global all_document_cosine_results
#global subsections_corpus
#global subsections_tfidf_results
#global subsections_cosine_results
#global entity_list_corpus
#global entity_list_tfidf_results
#global entity_list_cosine_results
#global one_pass_entity_list_corpus
#global one_pass_entity_list_tfidf_results
#global one_pass_entity_list_cosine_results
#
## initialize global variables
#everything_corpus = ""
#single_document_corpus = dict([])
#all_document_tfidf_results = {}
#all_document_cosine_results = {}
#subsections_corpus = dict([])
#subsections_tfidf_results = {}
#subsections_cosine_results = {}
#entity_list_corpus = dict([])
#entity_list_tfidf_results = {}
#entity_list_cosine_results = {}
#one_pass_entity_list_corpus = dict([])
#one_pass_entity_list_tfidf_results = {}
#one_pass_entity_list_cosine_results = {}


# ensure that there are no verified context errors for nltk
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')


'''
### Initial setup (Task 1) - unprocessed pages, subsections, and list of entities (except references) extracted ###

def setup():
    wikipedia = wikipediaapi.Wikipedia(
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI
    )
    global unprocessed_page
    unprocessed_page = dict([
        ('nature', wikipedia.page('Nature').text),
        ('pollution', wikipedia.page('pollution').text),
        ('sustainability', wikipedia.page('sustainability').text),
        ('environmentally_friendly', wikipedia.page('environmentally friendly').text)
    ])
    global page_subsections
    page_subsections = dict([
        ('nature', wikipedia.page('Nature').sections),
        ('pollution', wikipedia.page('pollution').sections),
        ('sustainability', wikipedia.page('sustainability').sections),
        ('environmentally_friendly', wikipedia.page('environmentally friendly').sections)
    ])
    # scrape clickable keywords without references
    # nature
    page_request = requests.get("https://en.wikipedia.org/wiki/Nature")
    beautiful_soup = BeautifulSoup(page_request.text, "html.parser")
    nature_entities_list = OrderedDict([])
    for link in beautiful_soup.find_all("a"):
        url = link.get("href", "")
        title = link.get("title", "")
        if title == "Balance of nature":
            nature_entities_list[title] = url
            break
        else:
            nature_entities_list[title] = url
    #pollution
    page_request = requests.get("https://en.wikipedia.org/wiki/Pollution")
    beautiful_soup = BeautifulSoup(page_request.text, "html.parser")
    pollution_entities_list = OrderedDict([])
    for link in beautiful_soup.find_all("a"):
        url = link.get("href", "")
        title = link.get("title", "")
        if url == "http://www.merriam-webster.com/dictionary/pollution":
            break
        else:
            pollution_entities_list[title] = url
    #sustainability
    page_request = requests.get("https://en.wikipedia.org/wiki/Sustainability")
    beautiful_soup = BeautifulSoup(page_request.text, "html.parser")
    sustainability_entities_list = OrderedDict([])
    for link in beautiful_soup.find_all("a"):
        url = link.get("href", "")
        title = link.get("title", "")
        if title == "Sustainable (song)":
            sustainability_entities_list[title] = url
            break
        else:
            sustainability_entities_list[title] = url
    #environmentally_friendly
    page_request = requests.get("https://en.wikipedia.org/wiki/Environmentally_friendly")
    beautiful_soup = BeautifulSoup(page_request.text, "html.parser")
    environmentally_friendly_entities_list = OrderedDict([])
    for link in beautiful_soup.find_all("a"):
        url = link.get("href", "")
        title = link.get("title", "")
        if title == "Sustainable products":
            environmentally_friendly_entities_list[title] = url
            break
        else:
            environmentally_friendly_entities_list[title] = url
    global page_entities_list
    page_entities_list = dict([
        ('nature', nature_entities_list),
        ('pollution', pollution_entities_list),
        ('sustainability', sustainability_entities_list),
        ('environmentally_friendly', environmentally_friendly_entities_list)
    ])

### Preprocessing and lemmatizing a single document ###

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

### Use recursion to get the wikipedia sections ###

def get_sections(sections, level=0):
    all_sections = ""
    for section in sections:
        all_sections = all_sections + section.title + " "
        get_sections(section.sections, level + 1)
    return all_sections

### Combining preprocessed and lemmatized documents into a corpus (Task 2A, 3A, 4A) ###

def corpus_creation(unprocessed_documents, type):
    corpus = ""
    if type == "pages":
        for key in unprocessed_documents:
            document = unprocessed_documents[key]
            corpus = corpus + preprocess_and_lemmatize(document)
            single_document_corpus[key]= corpus
        all_document_corpus = corpus
    elif type == "subsections":
        for key in page_subsections:
            for list in page_subsections[key]:
                all_sections = get_sections(list.sections)
                corpus = corpus + " " + preprocess_and_lemmatize(all_sections)
            subsections_corpus[key] = corpus
        all_document_corpus = corpus
    elif type == "keywords":
        for key in page_entities_list:
            for title in page_entities_list[key]:
                corpus = corpus + " " + preprocess_and_lemmatize(title)
            entity_list_corpus[key] = corpus
        all_document_corpus = corpus
    else:
        for key in all_one_pass_entity_categories:
            corpus = corpus + " " + preprocess_and_lemmatize(all_one_pass_entity_categories[key])
            one_pass_entity_list_corpus[key] = corpus

### TfidfVectorizer creation (Task 2B, 3B, 4B) ###

def vectorizer(document_1, document_2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([document_1, document_2])
    document_words = vectorizer.get_feature_names_out()

    # print the top 10 most used words from the tfidf results
    importance = np.argsort(np.asarray(vectors.sum(axis=0)).ravel())[::-1]
    tfidf_feature_names = np.array(vectorizer.get_feature_names_out())
    #print("MOST")
    #print(tfidf_feature_names[importance[:10]])

    # print the top 10 least used words from the tfidf results
    unimportance = np.argsort(np.asarray(vectors.sum(axis=0)).ravel())[::1]
    tfidf_feature_names = np.array(vectorizer.get_feature_names_out())
    #print("LEAST")
    #print(tfidf_feature_names[unimportance[:10]])

    dense = vectors.todense()
    dense_list = dense.tolist()
    calculated_table = pd.DataFrame(dense_list, columns=document_words)
    return calculated_table.T

### Calculate cosine similarity (Task 2C, 3C, 4C) ###

def calculate_cosine_similarity(document_1, document_2):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([document_1, document_2]).toarray()
    tfidfTran = TfidfTransformer()
    tfidfTran.fit(matrix)
    # added [0,1] so that the matrix is not printed
    # print(((vectors * vectors.T).A)[0,1]) # test that values are correct
    return cosine_similarity(matrix, matrix)[0,1]

### WuPalmer Wordnet calculations (Task 5A) ###

def calculate_wupalmer(word_1, word_2):
    syn_1 = wordnet.synsets(word_1)[0]
    syn_2 = wordnet.synsets(word_2)[0]
    wupalmer_similarity = syn_1.wup_similarity(syn_2)
    return wupalmer_similarity

### Scrape entity-categories from already found entity-categories (Task 6) ###

def entity_category_scraper(entity_category):
    page_request = requests.get("https://en.wikipedia.org" + entity_category)
    beautiful_soup = BeautifulSoup(page_request.text, "html.parser")
    page_entity_categories = ""
    for link in beautiful_soup.find_all("a"):
        page_entity_categories = page_entity_categories + " " + link.get("title", "")
    return page_entity_categories
'''

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

# Task 8
# word2vec

# Task 9
# news keywords

# Task 10 (repeat)
# repeat use of tfidf

# Task 12
# suggest a GUI