import wikipediaapi

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

    # TODO: Eliminate references from links
    global page_entities_list
    page_entities_list = dict([
        ('nature', wikipedia.page('Nature').links),
        ('pollution', wikipedia.page('pollution').links),
        ('sustainability', wikipedia.page('sustainability').links),
        ('environmentally_friendly', wikipedia.page('environmentally friendly').links)
    ])

### Preprocessing and lemmatization (Task 2A) ###

def preprocessing_and_lemmatization():
    print("preprocessing")

### Corpus creation (Task 2B) ###

def corpus_creation():
    print("corpus creation")

setup()
preprocessing_and_lemmatization()
corpus_creation()