'''
Implementation of the dataloader class & helper functions
'''
import os, csv, json
import math
import numpy as np
import string
from collections import Counter
from tqdm import tqdm
import random

'''
consts
'''
length = 10000

def get_dir(train_or_test):
    '''
    0 train 1 test
    '''
    if train_or_test == 0:
        return os.path.join('ag_news_csv', 'train.csv')
    else:
        return os.path.join('ag_news_csv', 'test.csv')    

def idf(corpus, words):
    '''
    corpus - list of str, each a documnt. 
    words - list of str, each an interest word.
    '''
    cnt = {}
    idf = {}
    for doc in tqdm(corpus):
        added = set()
        for boi in doc.split(' '):
            boi = ''.join(list(filter(lambda x: x.isalpha(), boi)))
            boi = boi.lower()
            if boi in words:
                if boi in added:
                    continue
                try:
                    cnt[boi] += 1
                except:
                    cnt[boi] = 1
                added.add(boi)
    for key in cnt.keys():
        idf[key] = math.log10((1 + len(corpus)) / (1 + cnt[key]))
    return idf

def get_stop_words(dir):
    with open(dir) as csv_file:
        reader = csv.reader(csv_file)
        stop_words = []
        for row in reader:
            for word in row:
                stop_words.append(word.strip())
    return stop_words

def get_idf(data_path, title_or_body):
    '''
    tob - 1 is title, 2 is body
    where vocab is produced
    '''
    stop_words = get_stop_words('stop_words.csv')
    corpus = []
    cnt = {}
    #words = set()
    with open(data_path) as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if title_or_body == 3:
                full_text = row[1] + '. ' + row[2]
            else:
                full_text = row[title_or_body]
            corpus.append(full_text)
            for word in full_text.split(' '):
                word = ''.join(list(filter(lambda x: x.isalpha(), word)))
                word = word.lower()
                if word in stop_words:
                    continue
                if len(word) <= 2:
                    continue
                #words.add(word)
                try:
                    cnt[word] += 1
                except:
                    cnt[word] = 1
    words = sorted(cnt.items(), key=lambda item:item[1], reverse=True)
    words = words[:length]
    words = [x[0] for x in words]
    idf_ = idf(corpus, words)
    vocab = {i:j for j, i in enumerate(words)}
    return vocab, idf_

def tf_idf(docs, vocab, idf):
    '''
    doc - n list
    '''
    out = np.zeros((len(docs), len(vocab)))
    #out = sp.csr_matrix((len(docs), len(vocab)), dtype=float)
    for i, doc in enumerate(docs):
        #words_clean = [i.strip() for i in docs.split(' ')]
        words = doc.split(' ')
        words = [(''.join(list(filter(lambda x: x.isalpha(), word)))).lower() for word in words]
        word_count = Counter(words)
        for word in words:
            if word not in vocab.keys():
                continue
            tfidf = (1+math.log10(word_count[word])) * idf[word]
            out[i, vocab[word]] = tfidf
    return out

class data_loader:
    def __init__(self, batch, dir, dir_for_savory, regen_stuff, shuffle = False) -> None:
        self.b = batch-1
        self.shuffle = shuffle
        if regen_stuff == 0:
            '''
            with open(os.path.join(dir_for_savory, 'vocab_body.json'), 'r') as f:
                self.vocab_body = json.load(f)
            with open(os.path.join(dir_for_savory, 'vocab_title.json'), 'r') as f:
                self.vocab_title = json.load(f) 
            with open(os.path.join(dir_for_savory, 'idf_body.json'), 'r') as f:
                self.idf_body = json.load(f)
            with open(os.path.join(dir_for_savory, 'idf_title.json'), 'r') as f:
                self.idf_title = json.load(f) 
            '''
            with open(os.path.join(dir_for_savory, 'vocab.json'), 'r') as f:
                self.vocab = json.load(f) 
            with open(os.path.join(dir_for_savory, 'idf.json'), 'r') as f:
                self.idf = json.load(f)
            print("loaded idfs and vocabs")
        else:
            '''
            self.vocab_title, self.idf_title = get_idf(dir, 1)
            self.vocab_body, self.idf_body = get_idf(dir, 2)
            print("generated idfs and vocabs. now saving.")
            with open(os.path.join(dir_for_savory, 'vocab_body.json'), 'w') as f:
                json.dump(self.vocab_body, f)
            with open(os.path.join(dir_for_savory, 'vocab_title.json'), 'w') as f:
                json.dump(self.vocab_title, f)
            with open(os.path.join(dir_for_savory, 'idf_body.json'), 'w') as f:
                json.dump(self.idf_body, f)
            with open(os.path.join(dir_for_savory, 'idf_title.json'), 'w') as f:
                json.dump(self.idf_title, f)
            '''
            self.vocab, self.idf = get_idf(dir, 3)
            with open(os.path.join(dir_for_savory, 'vocab.json'), 'w') as f:
                json.dump(self.vocab, f)
            with open(os.path.join(dir_for_savory, 'idf.json'), 'w') as f:
                json.dump(self.idf, f)
        #self.dims = len(self.vocab_title) + len(self.vocab_body)
        self.dims = len(self.vocab)
        self.b_num = 0        
        self.data_raw = []
        with open(dir) as csv_file:
            reader = csv.reader(csv_file)    
            for row in reader:
                self.data_raw.append((row[0], row[1], row[2]))
        self.shuffle_indice = [i for i in range(len(self.data_raw))]
        random.shuffle(self.shuffle_indice)
        print("loader initialized.")
    def __iter__(self):
        return self
    def __next__(self):
        if self.b_num * self.b < len(self.data_raw):
            if self.shuffle == True:
                selection_indice = self.shuffle_indice[self.b_num * self.b:self.b_num * self.b + self.b]
            else:
                selection_indice = range(self.b_num * self.b, self.b_num * self.b + self.b)
            self.b_num += 1
            labels = []
            '''
            docs_title = []
            docs_body  = []
            '''
            docs = []
            for i in selection_indice:
                try:
                    labels.append(self.data_raw[i][0])
                    '''
                    docs_title.append(self.data_raw[i][1]) 
                    docs_body.append(self.data_raw[i][2])
                    '''
                    docs.append(self.data_raw[i][1] + ". " + self.data_raw[i][2])
                except:
                    1
            '''
            tfidf_title = tf_idf(docs_title, self.vocab_title, self.idf_title)
            tfidf_body = tf_idf(docs_body, self.vocab_body, self.idf_body)
            '''
            tfidf = tf_idf(docs, self.vocab, self.idf)
            #return labels, np.hstack([tfidf_title, tfidf_body])
            return labels, tfidf
        else:
            StopIteration