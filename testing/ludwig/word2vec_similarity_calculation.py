# testing for Task 8 (word2vec task)

"""
The plan:
-   load the model
-   use the model on the four keywords
-   compute the similarity score and print it
"""

from gensim.models import KeyedVectors
from itertools import combinations
# import numpy as np

keywords = ['nature', 'pollution', 'sustainability', 'environmental']

# import the model (the model has to be downloaded beforehand)
# can be downloaded from here: https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
path = '~/Downloads/'
model = KeyedVectors.load_word2vec_format(path+'GoogleNews-vectors-negative300.bin', binary=True)

# calculate the cosine similarity measue for each keyword pair
print('===\nThe similarity scores using the pretrained word2vec model:')
for pair in list(combinations(keywords, 2)):
    score = model.similarity(pair[0], pair[1])
    print('->', pair, '=', score)


"""
observations:
-   "environmentally friendly" is not included in model; have to choose "environmental" or "environmentally",
    but the two words result in a significantly different score
"""


# old code (for testing)
# create vector for each keyword
# vectors = [None] * len(keywords)
# for i, keyword in enumerate(keywords):
#     vector = model[keyword]
#     print(keyword, 'is of type', type(vector), 'with length', len(vector))
#     vectors[i] = vector
#
# # testing if the score is really the same
# score = model.similarity('sustainability', 'environmental')
# print('similarity:', score)
# score_cosine_sim = np.dot(vectors[2], vectors[3]) / (np.linalg.norm(vectors[2]) * np.linalg.norm(vectors[3]))
# print('compared to the \"classic\" computation of cosine sim.:', score_cosine_sim)