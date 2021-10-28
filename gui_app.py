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
# window.grid_rowconfigure(0, weight=1)
# window.grid_columnconfigure(0, weight=1)

'''
# functions to execute actions

def button_clicked():
    # write text to the button; need to disable and enable afterwards
    txt_t1.configure(state='normal')
    txt_t1.delete('1.0', tk.END)
    txt_t1.insert(tk.END, '42')
    txt_t1.configure(state='disabled')


# set up labels

# task 1
label_t1 = tk.Label(window,
                    text=text_tasks[0],
                    font=default_label_font,
                    wraplength=label_wraplength,
                    justify=tk.LEFT)
label_t1.grid(row=0, column=0, sticky=tk.W)
label_t1.grid_rowconfigure(1, weight=1)
label_t1.grid_columnconfigure(1, weight=1)
btn_t1 = tk.Button(window, text=default_button_text, command=button_clicked)
btn_t1.grid(row=0, column=1)
#btn_t1.grid_rownconfigure(1, weight=1)
#btn_t1.grid_columnconfigure(1, weight=1)
txt_t1 = scrolledtext.ScrolledText(master=window, wrap=tk.WORD)
txt_t1.configure()
txt_t1.grid(row=1, column=0)

# task 2
label_t2 = tk.Label(window,
                    text=text_tasks[1],
                    font=default_label_font,
                    wraplength=label_wraplength,
                    justify=tk.LEFT)
label_t2.grid(row=2, column=0, sticky=tk.W)
label_t2.grid_rowconfigure(1, weight=1)
btn_t2 = tk.Button(window, text=default_button_text)
btn_t2.grid(row=2, column=1)

# task 3
label_t3 = tk.Label(window,
                    text=text_tasks[2],
                    font=default_label_font,
                    wraplength=label_wraplength,
                    justify=tk.LEFT)
label_t3.grid(row=3, column=0, sticky=tk.W)
label_t3.grid_rowconfigure(1, weight=1)
label_t3.grid_columnconfigure(1, weight=1)
btn_t3 = tk.Button(window, text=default_button_text)
btn_t3.grid(row=3, column=1)

# task 4
label_t4 = tk.Label(window,
                    text=text_tasks[3],
                    font=default_label_font,
                    wraplength=label_wraplength,
                    justify=tk.LEFT)
label_t4.grid(row=4, column=0, sticky=tk.W)
label_t4.grid_rowconfigure(1)
label_t4.grid_columnconfigure(1, weight=1)
btn_t4 = tk.Button(window, text=default_button_text)
btn_t4.grid(row=4, column=1)
'''


# now set up the connections between the buttons and the text boxes
def button_clicked(scrolled_text_widget):
    # write text to the button; need to disable and enable afterwards
    scrolled_text_widget.configure(state='normal')
    scrolled_text_widget.delete('1.0', tk.END)
    scrolled_text_widget.insert(tk.END, '42')
    #scrolled_text_widget.configure(state='disabled')

# now try to build the widgets using a for-loop
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
    btn = tk.Button(window, text=default_button_text, command=button_clicked(text_box))
    btn.grid(row=2*i, column=1, sticky=tk.W)
    btns.append(btn)


# test one case
#btns[0].configure(command=button_clicked(text_boxes[0]))

# for i, text in enumerate(text_tasks):
#
#     # link to functions
#     def button_clicked(scrolled_text_widget):


window.mainloop()
