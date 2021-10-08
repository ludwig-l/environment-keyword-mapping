# testing the functionalities

import wikipediaapi # https://wikipedia-api.readthedocs.io/en/latest/README.html
import nltk
import re
nltk.download('stopwords')


# just copied from: https://stackabuse.com/python-for-nlp-working-with-facebook-fasttext-library/
# observation: year numbers and citation codes remain after preprocessing
def preprocess_text(document):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(document))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    tokens = document.split()
    tokens = [nltk.stem.WordNetLemmatizer().lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if word not in set(nltk.corpus.stopwords.words('english'))]
    tokens = [word for word in tokens if len(word) > 3]

    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


# wiki_wiki = wikipediaapi.Wikipedia('en')
# nature = wiki_wiki.page('Nature').content
# print(type(nature))
# doc = nature.content # the whole wikipedia page as a string

# stopwords = list(set(nltk.corpus.stopwords.words('english')))
# print(stopwords)
# print('length:\n', stopwords)

# make it lowercase
#doc = doc.lower()

# set up wikipedia-api
wiki_wiki = wikipediaapi.Wikipedia(
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI # this formats the text nicely
)

# print out page content
doc = wiki_wiki.page('Nature').text
print(doc)


# seems like wikipedia-api makes life really easier since it already removes unecessary stuff :-)
# we only need to cut out the last sections with links and stuff (maybe)


# preprocess
doc_preprocessed = preprocess_text(doc)
print('preprocessed doc:\n', doc_preprocessed)

# tokenize
doc_tokenized = nltk.tokenize.word_tokenize(doc) # not quite sure what kind of tokenization to use (word or sentence?)
#print('tokenized:\n', doc_tokenized)


# next: use TfIdfVectorizer of each document and then calculate the cosine similarity


# ----------------------------------------


# only retrieve the sections from the wiki page
#doc_sections = wiki_wiki.page('Nature').sections
#print('--------\n---------')
#print('document sections:\n', doc_sections)