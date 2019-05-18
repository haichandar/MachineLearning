# -*- coding: utf-8 -*-
"""
Created on Thu May 16 11:26:48 2019

@author: Chandar_S
"""

import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import re

excel_data=pd.read_excel("Mytest_Clusters.xlsx", sheet_name="Sheet1")
ticket_data = excel_data.iloc[:,0:150] #Selecting the column that has text.


stop = set(stopwords.words('english'))

# custom words to ignore
custom_stop_words = ['tracke option', 'track options', 'service xcall', 'work note', 'service option']
for word in custom_stop_words:
    stop.add(word)    

exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

# Cleaning the text sentences so that punctuation marks, stop words & digits are removed
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    processed = re.sub(r"\d+","",normalized)
    y = processed.split()
    return y

New_Analysis_columnName = "Issue"
Analysis_ticket_columnName = "Ticket"

training_corpus=[]
training_description=[]
training_ticket_numbers=[]

for index,row in ticket_data.iterrows():
    
    line = ""
    if (row[New_Analysis_columnName] and str(row[New_Analysis_columnName]) != 'nan' ):
        line = str(row[New_Analysis_columnName])

        line = line.strip()
        cleaned = clean(line)

        ''' IF MANUAL CLASSFICATION IS AVAILABLE, PUT THEM INTO TRAINING, ELSE TESTING'''
        training_description.append(line)
        training_corpus.append(cleaned)
        training_ticket_numbers.append(row[Analysis_ticket_columnName])   

#print (training_description)
#%%
#from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
 
#print (common_texts)

"""
output:
[['human', 'interface', 'computer'], ['survey', 'user', 'computer', 'system', 'response', 'time'], ['eps', 'user', 'interface', 'system'], ['system', 'human', 'system', 'eps'], ['user', 'response', 'time'], ['trees'], ['graph', 'trees'], ['graph', 'minors', 'trees'], ['graph', 'minors', 'survey']]
"""
 
 
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(training_corpus)]
 
print (documents)
"""
output
[TaggedDocument(words=['human', 'interface', 'computer'], tags=[0]), TaggedDocument(words=['survey', 'user', 'computer', 'system', 'response', 'time'], tags=[1]), TaggedDocument(words=['eps', 'user', 'interface', 'system'], tags=[2]), TaggedDocument(words=['system', 'human', 'system', 'eps'], tags=[3]), TaggedDocument(words=['user', 'response', 'time'], tags=[4]), TaggedDocument(words=['trees'], tags=[5]), TaggedDocument(words=['graph', 'trees'], tags=[6]), TaggedDocument(words=['graph', 'minors', 'trees'], tags=[7]), TaggedDocument(words=['graph', 'minors', 'survey'], tags=[8])]
 
"""
 
model = Doc2Vec(documents, vector_size=300, window=2, min_count=1, workers=4)

#Persist a model to disk: 
from gensim.test.utils import get_tmpfile
fname = get_tmpfile("my_doc2vec_model")
 
print (fname)
#output: C:\Users\userABC\AppData\Local\Temp\my_doc2vec_model
 
#load model from saved file
model.save(fname)
model = Doc2Vec.load(fname)  
# you can continue training with the loaded model!
#If you’re finished training a model (=no more updates, only querying, reduce memory usage), you can do:
 
model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
 
#Infer vector for a new document:
#Here our text paragraph just 2 words
vector = model.infer_vector(['human', 'interface', 'computer'])
#print (vector)


#%%
import gensim.models as g
import codecs
 
model= "C:\\Users\\chandar_s\\AppData\\Local\\Temp\\my_doc2vec_model"
test_docs="data/test_docs.txt"
output_file="data/test_vectors.txt"
 
#inference hyper-parameters
start_alpha=0.01
infer_epoch=1000
 
#load model
m = g.Doc2Vec.load(model)
test_docs = [ x.strip().split() for x in codecs.open(test_docs, "r", "utf-8").readlines() ]

 
#infer test vectors
import numpy as np
vectors = np.zeros((len(test_docs), 300))
i = 0
for d in test_docs:
    vectors[i] =  m.infer_vector(d, alpha=start_alpha, steps=infer_epoch)
    i += 1

k_optimal = 3
from nltk.cluster import KMeansClusterer
import nltk
kclusterer = KMeansClusterer(k_optimal, distance=nltk.cluster.util.cosine_distance, repeats=25)
clusters = kclusterer.cluster(vectors, assign_clusters=True)

print (np.asarray(clusters))
