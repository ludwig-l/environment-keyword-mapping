from typing import OrderedDict
import wikipediaapi
from bs4 import BeautifulSoup
import requests
import re
import nltk
import ssl
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer

# global variables to store results

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

all_document_corpus = ""
all_document_tfidf_results = {}
all_document_cosine_results = {}

subsections_corpus = ""
subsections_tfidf_results = {}
subsections_cosine_results = {}

entity_list_corpus = ""
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

    # TAYLOR: wikipedia-api doesn't separate the references and is alphabetical, producing a messy list of links
    # TAYLOR: decided to use beautiful soup as an alternative

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

    for s in sections:
                   #print("%s:%s-%s" % ("*"*(level + 1), s.title, s.text[0:40]))
                   all_sections = all_sections + s.title + " " + s.text + " "
                   get_sections(s.sections, level + 1)

    return all_sections

### Combining preprocessed and lemmatized documents into a corpus (Task 2A, 3A, 4A) ###

def corpus_creation(unprocessed_documents, type):
    
    corpus = ""
    
    if type == "pages":
        for key in unprocessed_documents:
            document = unprocessed_documents[key]
            corpus = corpus + preprocess_and_lemmatize(document)
    elif type == "subsections":
        for key in page_subsections:
            for list in page_subsections[key]:
                all_sections = get_sections(list.sections)
                corpus = corpus + " " + preprocess_and_lemmatize(all_sections)
    else:
        for key in page_entities_list:
            for title in page_entities_list[key]:
                corpus = corpus + " " + preprocess_and_lemmatize(title)

    return corpus

### TfidfVectorizer creation (Task 2B, 3B, 4B) ###

def vectorizer():
    print("vectorizer")

    # found here: https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76
    #vectorizer = TfidfVectorizer()
    #vectors = vectorizer.fit_transform([documentA, documentB, documentC])
    #feature_names = vectorizer.get_feature_names()
    #dense = vectors.todense()
    #denselist = dense.tolist()
    #df = pd.DataFrame(denselist, columns=feature_names)

### Calculate cosine similarity (Task 2C, 3C, 4C) ###

def cosine_similarity():
    print ("cosine similarity")

    #https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
    #from scipy import spatial

    #dataSetI = [3, 45, 7, 2]
    #dataSetII = [2, 54, 13, 15]
    #result = 1 - spatial.distance.cosine(dataSetI, dataSetII)

### Wu and Palmer Wordnet calculations (Task 5A) ###

def wu_palm_calculations():
    print("wu and palmer wordnet calculations")

### Vector reproducing similarity and correlation between similarities (Task 5B) ###

def semantic_similarity_calculation():
    print("vector creation then semantic similarity calculation")


### main ###

setup()

# Task 2
#all_document_corpus = corpus_creation(unprocessed_page, "pages")
#print(all_document_corpus)
#vectorizer(all_document_corpus)
#cosine_similarity()

# Task 3
#subsection_corpus = corpus_creation(page_subsections, "subsections")
#print(subsections_corpus)
#vectorizer()
#cosine_similarity()

# Task 4
#entity_list_corpus = corpus_creation(page_entities_list, "keywords")
#print(entity_list_corpus)
#vectorizer()
#cosine_similarity()

# Task 5
#wu_palm_calculations()
#semantic_similarity_calculation()
