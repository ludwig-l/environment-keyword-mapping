# this will be the GUI application (hopefully ...)

import tkinter as tk
from tkinter import scrolledtext


# some definitions
default_button_text = 'Compute'
default_label_font = ('Arial Bold', 12)
text_tasks = [
    '1. Show Wikipedia pages for each keyword.',
    '2. Pre-process each dosument and calculate the cosine similarity measure of each document pair via Tf-Idf vectorizer approach.',
    '3. Show all titles of subsections of the Wikipedia pages and repeat the similarity score calculation.',
    '4. Show entity categories of the Wikipedia pages and repeat the similarity score calculation.',
    '5. Calculate Wu and Palmer semantic similarity between each keyword pari and calculate the correlation between the sementic similarity result and each of the three Wikipedia based similarities.',
    '6. Scrap content of each entity and retrieve all the entity-categories identified during this first exploration stage (first pass exploration).',
    '7. Repeat Tf-Idf vectorizer representation and recalculate the cosine similarity between individual words of the four keywords.',
    '8. Use a pre-trained word2vec model to represent each words and calculate the corresponding similarity.',
    '9. Retrieve articles from a news forum and retrieve information for different time periods. Display the word cloud representation for each document.',
    '10. Repeat the Tf-Idf based similarity calculus amoung each pair at each time.',
]
label_wraplength = 1000 # length for each line before the label text does a line break

# set up screen
window = tk.Tk()
window.title('Environment keyword mapping')
window.geometry('600x800')


# now set up the connections between the buttons and the text boxes
def button_clicked(scrolled_text_widget, idx):
    # write text to the button; need to disable and enable afterwards
    scrolled_text_widget.configure(state='normal')
    scrolled_text_widget.delete('1.0', tk.END)
    scrolled_text_widget.insert(tk.END, '42')
    scrolled_text_widget.configure(state='disabled')
    print('This is button no.', idx)

# build the widgets using a for loop and append them to the for loop
labels = []
btns = []
text_boxes = []
for i, text in enumerate(text_tasks):

    # set up label, button and text box
    label = tk.Label(window,
                     text=text,
                     font=default_label_font,
                     wraplength=label_wraplength,
                     justify=tk.LEFT)
    label.grid(row=2*i, column=0, sticky=tk.W)
    labels.append(label)
    text_box = scrolledtext.ScrolledText(wrap=tk.WORD, state='disabled')
    text_box.grid(row=2*i+1, column=0)
    text_boxes.append(text_box)
    btn = tk.Button(window,
                    text=default_button_text,
                    command=lambda scrolled_text_obj=text_box, idx=i:
                    button_clicked(scrolled_text_obj, idx))
    btn.grid(row=2*i, column=1, sticky=tk.W)
    btns.append(btn)


window.mainloop()
