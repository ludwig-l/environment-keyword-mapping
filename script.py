import array
from typing import OrderedDict
from numpy import single
import wikipediaapi
from bs4 import BeautifulSoup
import requests
import re
import nltk
import ssl
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import pandas as pd
from itertools import combinations
from scipy import spatial
from scipy.stats import pearsonr

# global variables to store results

global everything_corpus

global all_document_corpus
global all_document_tfidf_results
global all_document_cosine_results

global subsections_corpus
global subsections_tfidf_results
global subsections_cosine_results

global entity_list_corpus
global entity_list_tfidf_results
global entity_list_cosine_results

# initialize global variables

everything_corpus = ""

single_document_corpus = dict([])
all_document_tfidf_results = {}
all_document_cosine_results = {}

subsections_corpus = dict([])
subsections_tfidf_results = {}
subsections_cosine_results = {}

entity_list_corpus = dict([])
entity_list_tfidf_results = {}
entity_list_cosine_results = {}

# ensure that there are no verified context errors for nltk

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')

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
    elif type == "subsections":
        for key in page_subsections:
            for list in page_subsections[key]:
                all_sections = get_sections(list.sections)
                corpus = corpus + " " + preprocess_and_lemmatize(all_sections)
            subsections_corpus[key] = corpus
    else:
        for key in page_entities_list:
            for title in page_entities_list[key]:
                corpus = corpus + " " + preprocess_and_lemmatize(title)
            entity_list_corpus[key] = corpus
    
    all_document_corpus = corpus

### TfidfVectorizer creation (Task 2B, 3B, 4B) ###

def vectorizer(document_1, document_2):

    # found here: https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(document_1, document_2)
    document_words = vectorizer.get_feature_names_out()
    dense = vectors.todense()
    dense_list = dense.tolist()
    calculated_table = pd.DataFrame(dense_list, columns=document_words)
    return calculated_table

### Calculate cosine similarity (Task 2C, 3C, 4C) ###

def cosine_similarity(document_1, document_2):

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([document_1, document_2])
    return ((vectors * vectors.T).A)[0,1]

### WuPalmer Wordnet calculations (Task 5A) ###

def calculate_wupalmer(word_1, word_2):
    print(word_1, word_2)
    syn_1 = wordnet.synsets(word_1)[0]
    syn_2 = wordnet.synsets('environmentally')[0]
    wupalmer_similarity = syn_1.wup_similarity(syn_2)
    return wupalmer_similarity

### Scrape entity-categories from already found entity-categories (Task 6) ###

def entity_category_scraper(entity_category):
    page_request = requests.get("https://en.wikipedia.org/wiki/" + entity_category)
    beautiful_soup = BeautifulSoup(page_request.text, "html.parser")
    page_entity_categories = []
    for link in beautiful_soup.find_all("a"):
        page_entity_categories.append(link.get("title", ""))
    return page_entity_categories

### main ###

# Task 1: Get unprocessed pages, subsections, and list of entities (clickable keywords except reference list)
setup()

# Task 2: Create combined corpus, as well as separate corpuses to do TFIDF and cosine similarity
corpus_creation(unprocessed_page, "pages")
#print(all_document_corpus['nature'])
corpus_creation(page_subsections, "subsections")
#print(subsections_corpus['nature'])
corpus_creation(page_entities_list, "keywords")
#print(entity_list_corpus['nature'])
#print(everything_corpus)   # combination of all documents into one corpus
all_cosine_results = array.array('d', [])
for pair in list(combinations(list(single_document_corpus), 2)):
    tfidf_results = vectorizer([single_document_corpus[pair[0]]], [single_document_corpus[pair[1]]])
    #print(tfidf_results)
    all_cosine_results.append(cosine_similarity(single_document_corpus[pair[0]], single_document_corpus[pair[1]]))
#print(all_cosine_results)

# Task 3: Repeat but with the titles of subsections
for pair in list(combinations(list(subsections_corpus), 2)):
    tfidf_results = vectorizer([subsections_corpus[pair[0]]], [subsections_corpus[pair[1]]])
    #print(tfidf_results)
    cosine_results = cosine_similarity(subsections_corpus[pair[0]], subsections_corpus[pair[1]])
    #print(cosine_results)

# Task 4: Repeat but with the entity-categories
for pair in list(combinations(list(entity_list_corpus), 2)):
    tfidf_results = vectorizer([entity_list_corpus[pair[0]]], [entity_list_corpus[pair[1]]])
    #print(tfidf_results)
    cosine_results = cosine_similarity(entity_list_corpus[pair[0]], entity_list_corpus[pair[1]])
    #print(cosine_results)

# Task 5: Calculate wu and Palmer WordNet semantic similarity, write a vector representing that similarity,
# and calculate the correlation between the semantic similarity and the wikipedia based similarity
pair_words = ['nature', 'pollution', 'sustainability', 'environmentally']
all_wupalmer_results = array.array('d', [])
for pair in combinations(pair_words, 2):
    all_wupalmer_results.append(calculate_wupalmer(pair[0], pair[1]))
wu_wiki_correlation = pearsonr(all_wupalmer_results, all_cosine_results)
# first value is the pearson's correlation coefficient, second value is the two-tailed p-value
print(wu_wiki_correlation)

# Task 6: Scrape content of each entity and retrieve all clickable keywords identified


entity_category_scraper


# Task 7
# repeat on the stuff from task 6

# Task 8
# word2vec

# Task 9
# news keywords

# Task 10 (repeat)
# repeat use of tfidf

# Task 12
# suggest a GUI