# this will be the GUI application (hopefully ...)

import tkinter as tk
from tkinter import scrolledtext
from utils import Utils
import array
from itertools import combinations


# some definitions
default_button_text = 'Compute'
default_label_font = ('Arial Bold', 11)
text_tasks = [
    '1. Show Wikipedia pages for each keyword.',
    '2. Pre-process each dosument and calculate the cosine similarity measure of each document pair via Tf-Idf vectorizer approach.',
    '3. Show all titles of subsections of the Wikipedia pages and repeat the similarity score calculation.',
    '4. Show entity categories of the Wikipedia pages and repeat the similarity score calculation.',
    '5. Calculate Wu and Palmer semantic similarity between each keyword pari and calculate the correlation between the sementic similarity result and each of the three Wikipedia based similarities.',
    '6. Scrap content of each entity and retrieve all the entity-categories identified during this first exploration stage (first pass exploration).',
    '7. Repeat Tf-Idf vectorizer representation and recalculate the cosine similarity between individual words of the four keywords.',
    '8. Use a pre-trained word2vec model to represent each words and calculate the corresponding similarity (place the model file inside your local user\'s downloads folder).',
    '9. Retrieve articles from a news forum and retrieve information for different time periods. Display the word cloud representation for each document.',
    '10. Repeat the Tf-Idf based similarity calculus amoung each pair at each time.',
]
label_wraplength = 850  # length for each line before the label text does a line break
textbox_width = 117
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

    # check for page the button is belonging to and execute the respective function then

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

        scrolled_text_widget.insert(tk.END, list(combinations(list(obj.single_document_corpus), 2)))
        scrolled_text_widget.insert(tk.END, '\n')
        scrolled_text_widget.insert(tk.END, all_cosine_results)

    # task 3
    if idx == 2:
        scrolled_text_widget.insert(tk.END, obj.page_subsections)

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
        scrolled_text_widget.insert(tk.END, obj.page_entities_list)

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

        scrolled_text_widget.insert(tk.END, list(combinations(pair_words, 2)))
        scrolled_text_widget.insert(tk.END, '\n')
        scrolled_text_widget.insert(tk.END, all_wupalmer_results)


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
