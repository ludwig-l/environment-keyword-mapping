# class for providing all necessary functions and variables for storing the data of interest

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


class Utils:
    def __init__(self):
        # something something ...
        self.everything_corpus = ""
        self.single_document_corpus = dict([])
        self.all_document_tfidf_results = {}
        self.all_document_cosine_results = {}
        self.subsections_corpus = dict([])
        self.subsections_tfidf_results = {}
        self.subsections_cosine_results = {}
        self.entity_list_corpus = dict([])
        self.entity_list_tfidf_results = {}
        self.entity_list_cosine_results = {}
        self.one_pass_entity_list_corpus = dict([])
        self.one_pass_entity_list_tfidf_results = {}
        self.one_pass_entity_list_cosine_results = {}

    # function definitions
    ### Initial setup (Task 1) - unprocessed pages, subsections, and list of entities (except references) extracted ###
    def setup(self):
        wikipedia = wikipediaapi.Wikipedia(
            language='en',
            extract_format=wikipediaapi.ExtractFormat.WIKI
        )
        self.unprocessed_page = dict([
            ('nature', wikipedia.page('Nature').text),
            ('pollution', wikipedia.page('pollution').text),
            ('sustainability', wikipedia.page('sustainability').text),
            ('environmentally_friendly', wikipedia.page('environmentally friendly').text)
        ])
        self.page_subsections = dict([
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
        self.page_entities_list = dict([
            ('nature', nature_entities_list),
            ('pollution', pollution_entities_list),
            ('sustainability', sustainability_entities_list),
            ('environmentally_friendly', environmentally_friendly_entities_list)
        ])


    ### Preprocessing and lemmatizing a single document ###
    def preprocess_and_lemmatize(self, document):
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
    def get_sections(self, sections, level=0):
        all_sections = ""
        for section in sections:
            all_sections = all_sections + section.title + " "
            self.get_sections(section.sections, level + 1)
        return all_sections


    ### Combining preprocessed and lemmatized documents into a corpus (Task 2A, 3A, 4A) ###
    def corpus_creation(self, unprocessed_documents, type):
        corpus = ""
        if type == "pages":
            for key in unprocessed_documents:
                document = unprocessed_documents[key]
                corpus = corpus + self.preprocess_and_lemmatize(document)
                self.single_document_corpus[key]= corpus
            all_document_corpus = corpus
        elif type == "subsections":
            for key in self.page_subsections:
                for list in self.page_subsections[key]:
                    all_sections = self.get_sections(list.sections)
                    corpus = corpus + " " + self.preprocess_and_lemmatize(all_sections)
                self.subsections_corpus[key] = corpus
            all_document_corpus = corpus
        elif type == "keywords":
            for key in self.page_entities_list:
                for title in self.page_entities_list[key]:
                    corpus = corpus + " " + self.preprocess_and_lemmatize(title)
                self.entity_list_corpus[key] = corpus
            all_document_corpus = corpus
        else:
            for key in self.all_one_pass_entity_categories:
                corpus = corpus + " " + self.preprocess_and_lemmatize(self.all_one_pass_entity_categories[key])
                self.one_pass_entity_list_corpus[key] = corpus


    ### TfidfVectorizer creation (Task 2B, 3B, 4B) ###
    def vectorizer(self, document_1, document_2):
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
    def calculate_cosine_similarity(self, document_1, document_2):
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform([document_1, document_2]).toarray()
        tfidfTran = TfidfTransformer()
        tfidfTran.fit(matrix)
        # added [0,1] so that the matrix is not printed
        # print(((vectors * vectors.T).A)[0,1]) # test that values are correct
        return cosine_similarity(matrix, matrix)[0,1]


    ### WuPalmer Wordnet calculations (Task 5A) ###
    def calculate_wupalmer(self, word_1, word_2):
        syn_1 = wordnet.synsets(word_1)[0]
        syn_2 = wordnet.synsets(word_2)[0]
        wupalmer_similarity = syn_1.wup_similarity(syn_2)
        return wupalmer_similarity


    ### Scrape entity-categories from already found entity-categories (Task 6) ###
    def entity_category_scraper(self, entity_category):
        page_request = requests.get("https://en.wikipedia.org" + entity_category)
        beautiful_soup = BeautifulSoup(page_request.text, "html.parser")
        page_entity_categories = ""
        for link in beautiful_soup.find_all("a"):
            page_entity_categories = page_entity_categories + " " + link.get("title", "")
        return page_entity_categories