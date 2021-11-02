# this will be the GUI application (hopefully ...)

import tkinter as tk
from tkinter import scrolledtext
from utils import Utils
import array
from itertools import combinations
from scipy.stats import pearsonr


# some definitions
default_button_text = 'Compute'
default_label_font = ('Arial Bold', 11)
text_tasks = [
    '1. Show a preview of each page\'s corresponding Wikipedia page.',
    '2. Pre-process each document and calculate the cosine similarity of each document pair using tf-idf.',
    '3. Repeat the similarity score calculation on the Wikipedia subsection titles.',
    '4. Repeat the similarity score calculation on the Wikipedia entity categories.',
    '5. Calculate Wu and Palmer semantic similarity between each keyword pair and calculate the correlation between that result and the Wikipedia cosine similarity result.',
    '6. Scrape the content of each entity and retrieve all the entity-categories identified using first pass exploration (This task can take several minutes to complete).',
    '7. Perform tf-idf and cosine similarity on the scraped entity list for each keyword Wikipedia page.',
    '8. Use a pre-trained word2vec model to represent each keyword and calculate the corresponding similarity (place the model file inside your local user\'s downloads folder).',
    '9. Retrieve articles from a news forum and retrieve information for different time periods. Display the word cloud representation for each document.',
    '10. Repeat the tf-idf based similarity calculus between each pair at each time.',
]
label_wraplength = 650  # length for each line before the label text does a line break
textbox_width = 100
textbox_height = 9

# set up screen
window = tk.Tk()
window.title('Environment keyword mapping')
window.geometry('1980x1080')

# now set up the connections between the buttons and the text boxes
def button_clicked(scrolled_text_widget, idx):
    # write text to the button; need to disable and enable afterwards
    scrolled_text_widget.configure(state='normal')
    scrolled_text_widget.delete('1.0', tk.END)

    # check for the page the button belongs to and execute the respective function then

    # task 1
    if idx == 0:
        scrolled_text_widget.insert(
            tk.END,
            'Script will fetch the wikipedia pages now...\nContent of the wikipedia pages:\n\n'
        )
        obj.setup()
        scrolled_text_widget.insert(tk.END, obj.unprocessed_page['nature'])
        scrolled_text_widget.insert(tk.END, '\n\n----------------------------------------\n\n')
        scrolled_text_widget.insert(tk.END, obj.unprocessed_page['pollution'])
        scrolled_text_widget.insert(tk.END, '\n\n----------------------------------------\n\n')
        scrolled_text_widget.insert(tk.END, obj.unprocessed_page['sustainability'])
        scrolled_text_widget.insert(tk.END, '\n\n----------------------------------------\n\n')
        scrolled_text_widget.insert(tk.END, obj.unprocessed_page['environmentally_friendly'])

    # task 2
    if idx == 1:
        obj.corpus_creation(obj.unprocessed_page, 'pages')
        obj.corpus_creation(obj.page_subsections, 'subsections')
        obj.corpus_creation(obj.page_entities_list, 'keywords')

        all_cosine_results = array.array('d', [])
        for pair in list(combinations(list(obj.single_document_corpus), 2)):
            obj.tfidf_results = obj.vectorizer(obj.single_document_corpus[pair[0]],
                                        obj.single_document_corpus[pair[1]])
            all_cosine_results.append(
                obj.calculate_cosine_similarity(obj.single_document_corpus[pair[0]],
                                                obj.single_document_corpus[pair[1]]))
            obj.cosine_result = obj.calculate_cosine_similarity(
                obj.single_document_corpus[pair[0]],
                obj.single_document_corpus[pair[1]])

        # assign value to helper variable in order to use this value later on again in this script
        obj.all_cosine_results = all_cosine_results

        scrolled_text_widget.insert(tk.END, list(combinations(list(obj.single_document_corpus), 2)))
        scrolled_text_widget.insert(tk.END, '\n')
        scrolled_text_widget.insert(tk.END, all_cosine_results)

    # task 3
    if idx == 2:
        #scrolled_text_widget.insert(tk.END, obj.page_subsections)

        cosine_results_list = []
        for pair in list(combinations(list(obj.subsections_corpus), 2)):
            tfidf_results = obj.vectorizer(obj.subsections_corpus[pair[0]], obj.subsections_corpus[pair[1]])
            cosine_results = obj.calculate_cosine_similarity(
                obj.subsections_corpus[pair[0]], obj.subsections_corpus[pair[1]])
            cosine_results_list.append(cosine_results)

        scrolled_text_widget.insert(tk.END, '\n\n----------------------------------------\n\n')
        scrolled_text_widget.insert(tk.END, 'Cosine similarity results:\n\n')
        scrolled_text_widget.insert(tk.END, list(combinations(list(obj.subsections_corpus), 2)))
        scrolled_text_widget.insert(tk.END, '\n')
        scrolled_text_widget.insert(tk.END, cosine_results_list)

    # task 4
    if idx == 3:
        #scrolled_text_widget.insert(tk.END, obj.page_entities_list)

        cosine_results_list = []
        for pair in list(combinations(list(obj.entity_list_corpus), 2)):
            tfidf_results = obj.vectorizer(obj.entity_list_corpus[pair[0]],
                                           obj.entity_list_corpus[pair[1]])
            cosine_results = obj.calculate_cosine_similarity(
                obj.entity_list_corpus[pair[0]],
                obj.entity_list_corpus[pair[1]])
            cosine_results_list.append(cosine_results)

        scrolled_text_widget.insert(tk.END, '\n\n----------------------------------------\n\n')
        scrolled_text_widget.insert(tk.END, 'Cosine similarity results:\n\n')
        scrolled_text_widget.insert(tk.END, list(combinations(list(obj.entity_list_corpus), 2)))
        scrolled_text_widget.insert(tk.END, '\n')
        scrolled_text_widget.insert(tk.END, cosine_results_list)

    # task 5
    if idx == 4:
        pair_words = ['nature', 'pollution', 'sustainability', 'environment']
        all_wupalmer_results = array.array('d', [])
        for pair in combinations(pair_words, 2):
            all_wupalmer_results.append(obj.calculate_wupalmer(pair[0], pair[1]))

        wu_wiki_correlation = pearsonr(all_wupalmer_results, obj.all_cosine_results)

        scrolled_text_widget.insert(tk.END, 'Wu and Palmer semantic similarity results:\n\n')
        scrolled_text_widget.insert(tk.END, list(combinations(pair_words, 2)))
        scrolled_text_widget.insert(tk.END, '\n')
        scrolled_text_widget.insert(tk.END, all_wupalmer_results)
        scrolled_text_widget.insert(tk.END, '\n\n----------------------------------------\n\n')
        scrolled_text_widget.insert(tk.END, 'Correlation between the sementic similarity result and each of the three Wikipedia based similarities:\n')
        scrolled_text_widget.insert(tk.END, '\n')
        scrolled_text_widget.insert(tk.END, wu_wiki_correlation)

    # task 6
    if idx == 5:
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

        scrolled_text_widget.insert(tk.END, obj.all_one_pass_entity_categories)

    # task 7
    if idx == 6:
        cosine_results_list = []

        obj.corpus_creation(obj.all_one_pass_entity_categories, "keywords_2")
        for pair in list(combinations(list(obj.one_pass_entity_list_corpus), 2)):
            tfidf_results = obj.vectorizer(obj.one_pass_entity_list_corpus[pair[0]],
                                    obj.one_pass_entity_list_corpus[pair[1]])
            cosine_results = obj.calculate_cosine_similarity(
                obj.one_pass_entity_list_corpus[pair[0]],
                obj.one_pass_entity_list_corpus[pair[1]])
            cosine_results_list.append(cosine_results)

        scrolled_text_widget.insert(tk.END, 'Cosine similarity results:\n\n')
        scrolled_text_widget.insert(tk.END, list(combinations(list(obj.one_pass_entity_list_corpus), 2)))
        scrolled_text_widget.insert(tk.END, '\n')
        scrolled_text_widget.insert(tk.END, cosine_results_list)

    # task 8
    if idx == 7:
        word2vec_model_file_path = '~/Downloads/GoogleNews-vectors-negative300.bin'
        keywords = ['nature', 'pollution', 'sustainability', 'environmental']
        word2vec_scores = obj.calc_word2vec_scores(keywords, word2vec_model_file_path)

        scrolled_text_widget.insert(tk.END, word2vec_scores)

    # task 9
    if idx == 8:
        n_news_forum_articles = 250

        scrolled_text_widget.insert(tk.END, 'Start retrieving articles now ...')
        scrolled_text_widget.insert(tk.END, '\n\n')

        obj.retrieve_articles(n_news_forum_articles, obj.news_forum_data)

        scrolled_text_widget.insert(
            tk.END,
            'Word cloud representations will open in a separate window but will also be stored as a file in the main directory.'
        )

        obj.display_word_cloud_represenations(obj.news_forum_data)

    # task 10
    if idx == 9:
        obj.calc_tfidf_scores_news_forum(obj.news_forum_data)

        tmp = obj.calc_tfidf_scores_news_forum(obj.news_forum_data)
        for data in tmp:
            scrolled_text_widget.insert(tk.END, data)
            scrolled_text_widget.insert(tk.END, '\n')


    scrolled_text_widget.configure(state='disabled')


# create on object for all the utility functions
obj = Utils()

# build the widgets using a for loop and append them to the for loop
labels = []
btns = []
text_boxes = []
for i, text in enumerate(text_tasks):

    # select current column (will be 0 for the first 5 elements and 2 for the other 5 elements)
    curr_col = 0
    if i >= 5: curr_col = 2
    # select current row (will be in the same column for numbers 4-9)
    curr_row = i
    if i >= 5: curr_row -= 5

    # set up label, button and text box
    label = tk.Label(window,
                     text=text,
                     font=default_label_font,
                     wraplength=label_wraplength,
                     justify=tk.LEFT)
    label.grid(row=2*curr_row, column=curr_col, sticky=tk.W)
    labels.append(label)
    window.grid_columnconfigure(curr_col)
    text_box = scrolledtext.ScrolledText(window,
                                         wrap=tk.WORD,
                                         state='disabled',
                                         height=textbox_height,
                                         width=textbox_width)
    text_box.grid(row=2*curr_row+1, column=curr_col, columnspan=2)
    text_boxes.append(text_box)
    btn = tk.Button(window,
                    text=default_button_text,
                    command=lambda scrolled_text_obj=text_box, idx=i:
                    button_clicked(scrolled_text_obj, idx))
    btn.grid(row=2*curr_row, column=curr_col+1, sticky=tk.W)
    btns.append(btn)


window.mainloop()
