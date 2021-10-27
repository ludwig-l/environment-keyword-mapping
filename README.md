# Environment keyword mapping

Review selected environmental topics according to Wikipedia-based description.

# How to run the script

1. Install the required libraries. We recommend to create a virtual environment and then run `pip install requirements.txt`.
2. Currently, there is not one script for tasks. Task 1-7 can be found in `script.py`, task 8 can be found in `word2vec_similarity_calculation.py` and tasks 9 and 10 can be found in `news_forum_retrieval.py`.

## Some more notes

* The script uses artical retrieval from the New York Times. Therefore, a valid API key is needed which can be obtained from [here](https://developers.nytimes.com/). Create a file named `nytimes_api_key.txt` and place it in the same folder. The script will read in the key from this file.
* In the file `gui_app.py` the GUI implementation of the script can be found, but currently this is still under development.
* For the word2vec part of the program (task 8) one needs to download the pre-trained model. In the respective part of the script there is a hint on where to download the model.
