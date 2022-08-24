import json
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import time
import pickle

np.random.seed(0)
wiki_data = []
with open("enwiki_20220201.json", "r") as f:
    for each_line in f:
        record = json.loads(each_line)
        wiki_data.append(record)
l = len(wiki_data)
sentences = []
words = []
time0 = time.time()
for i in range(len(wiki_data)):
    sentences.append(sent_tokenize(wiki_data[0]['text']))
    words.append(word_tokenize(wiki_data[0]['text']))
    if (i + 1) % 100 == 0:
        print("%d/%d" % (i + 1, l), time.time() - time0)
with open('tokens.pk') as f:
    pickle.dump(dict(sentences=sentences, words=words), f)
