# Environment keyword mapping

Review selected environmental topics according to Wikipedia-based description.

# How to run the script

1. Install the required libraries. We recommend to create a virtual environment and then run `pip install requirements.txt`.
2. All tasks can be found in `script.py`, they are all executed one by one.
3. One can run the script using a GUI. Therefore, just execute the file `gui_app.py`. The GUI uses *Tkinter*.

## Some more notes

* The script uses artical retrieval from the New York Times. Therefore, a valid API key is needed which can be obtained from [here](https://developers.nytimes.com/). Create a file named `nytimes_api_key.txt` and place it in the same folder. The script will read in the key from this file.
* For the word2vec part of the program (task 8) one needs to download the pre-trained model. In the respective part of the script there is a hint on where to download the model.
