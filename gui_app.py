# this will be the GUI application (hopefully ...)

import tkinter as tk
from tkinter import scrolledtext


# some definitions
default_button_text = 'Compute'
default_label_font = ('Arial Bold', 20)

# set up screen
window = tk.Tk()
window.title('Environment keyword mapping')
window.geometry('600x800')
# window.grid_rowconfigure(0, weight=1)
# window.grid_columnconfigure(0, weight=1)


# functions to execute actions

def button_clicked():
    # write text to the button; need to disable and enable afterwards
    txt_t1.configure(state='normal')
    txt_t1.delete('1.0', tk.END)
    txt_t1.insert(tk.END, '42')
    txt_t1.configure(state='disabled')


# set up labels

# task 1
label_t1 = tk.Label(window, text='First task', font=default_label_font)
label_t1.grid(row=0, column=0)
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
label_t2 = tk.Label(window, text='Second task', font=default_label_font)
label_t2.grid(row=2, column=0)
label_t2.grid_rowconfigure(1, weight=1)
label_t2.grid_columnconfigure(1, weight=1)
btn_t2 = tk.Button(window, text=default_button_text)
btn_t2.grid(row=2, column=1)

# task 3
label_t3 = tk.Label(window, text='Third task', font=default_label_font)
label_t3.grid(row=3, column=0)
label_t3.grid_rowconfigure(1, weight=1)
label_t3.grid_columnconfigure(1, weight=1)
btn_t3 = tk.Button(window, text=default_button_text)
btn_t3.grid(row=3, column=1)

# task 4
label_t4 = tk.Label(window,text='Fourth task', font=default_label_font)
label_t4.grid(row=4, column=0)
label_t4.grid_rowconfigure(1)
label_t4.grid_columnconfigure(1, weight=1)
btn_t4 = tk.Button(window, text=default_button_text)
btn_t4.grid(row=4, column=1)

window.mainloop()
