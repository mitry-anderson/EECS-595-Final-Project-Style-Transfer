from nltk.lm import Vocabulary
from nltk.lm.models import KneserNey, KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk import ngrams
from nltk import FreqDist
from nltk import KneserNeyProbDist

ngram_order = 3

id_to_label = ['adventure', 
                    'belles_lettres', 
                    'editorial', 
                    'fiction', 
                    'government', 
                    'hobbies', 
                    'humor', 
                    'learned', 
                    'lore', 
                    'mystery', 
                    'news', 
                    'religion', 
                    'reviews', 
                    'romance', 
                    'science_fiction'
                    ]
stage = 'train'

train_sentences = []
train_labels = []
for i in range(len(id_to_label)):
    with open(f'dataset/{id_to_label[i]}_{stage}.txt') as file:
        for sentence in file:
            train_sentences.append(sentence)
            train_labels.append(i)

test_sentences = []
test_labels = []
for i in range(len(id_to_label)):
    with open(f'dataset/{id_to_label[i]}_{stage}.txt') as file:
        for sentence in file:
            test_sentences.append(sentence)
            test_labels.append(i)

train_data,  vocab_data = padded_everygram_pipeline(ngram_order, train_sentences)

kn = KneserNeyInterpolated(ngram_order)

kn.fit(train_data, vocab_data)

print(len(kn.vocab))
ppw = 0
for sentence in test_sentences:
    temp = 1
    for i in range(len(sentence)):
        temp *= 1/(kn.score(sentence[i], sentence[i-3:i-1]))
    temp
    










