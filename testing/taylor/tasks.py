from typing import OrderedDict
import wikipediaapi
from bs4 import BeautifulSoup
import requests
import re
import nltk
import ssl
from sklearn.feature_extraction.text import TfidfVectorizer

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

### Preprocessing, lemmatizing, and combining into a corpus (Task 2A, 3A, 4A) ###

def corpus_creation(unprocessed_documents):
    
    corpus = ""
    for key in unprocessed_documents:
        document = unprocessed_documents[key]

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
        # remove numbers that are out of year range 1900-2050
        document = re.sub(r'')

        print(document)

        # lemmatize

        document_tokens = document.split()
        #document_tokens = [nltk.stem.WordNetLemmatizer().lemmatize(word) for word in document_tokens]
        #document_tokens = [word for word in document_tokens if word not in (set(nltk.corpus.stopwords.words('english') and set(nltk.corpus.stopwords.word('french')))) ]
        #document_tokens = [word for word in document_tokens if len(word) > 2]
        
        print(document_tokens)
        break

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
corpus_creation(unprocessed_page)
#vectorizer()
#cosine_similarity()

# Task 3
#corpus_creation(page_subsections)
#vectorizer()
#cosine_similarity()

# Task 4
#corpus_creation(page_entities_list)
#vectorizer()
#cosine_similarity()

# Task 5
#wu_palm_calculations()
#semantic_similarity_calculation()
