# -*- coding: utf-8 -*-
''' This program takes a excel sheet as input where each row in first column of sheet represents a document.  '''

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import string
import re

# YOU NEED TO DO THIS FIRST TIME TO DOWNLOAD FEW CORPORA FOR TEXT ANALYSIS
#import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')

''' HYPER PARAMETERS '''
input_file = 'T&ADataForAnalysis'
data=pd.read_excel(input_file +'.xlsx', sheet_name="BaseData") #Include your data file instead of data.xlsx
ticket_data = data.iloc[:,0:30] #Selecting the column that has text.
Analysis_primary_columnName = 'Description (Customer visible)'
Analysis_secondary_columnName = 'Short description'
Analysis_Result_columnName = 'SerialNumber'
Analysis_ticket_columnName  = 'Number'
num_clusters = 75 #Change it according to your data.
''' HYPER PARAMETERS ''' 

stop = set(stopwords.words('english'))
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

#Converting the column of data from excel sheet into a list of documents, where each document corresponds to a group of words.
training_corpus=[]
training_description=[]
testing_corpus=[]
testing_description=[]
training_ticket_numbers=[]
testing_ticket_numbers=[]
training_output_category=[]
for index,row in ticket_data.iterrows():
    
    line = ""
    if (row[Analysis_primary_columnName] and str(row[Analysis_primary_columnName]) != 'nan' ):
        line = str(row[Analysis_primary_columnName])
    else:
        line = str(row[Analysis_secondary_columnName])

    line = line.strip()
    cleaned = clean(line)
    cleaned = ' '.join(cleaned)

    ''' IF MANUAL CLASSFICATION IS AVAILABLE, PUT THEM INTO TRAINING, ELSE TESTING'''
    if (str(row[Analysis_Result_columnName]) != 'nan'):
        training_description.append(line)
        training_corpus.append(cleaned)
        # Add ticket number for indexing
        training_ticket_numbers.append(row[Analysis_ticket_columnName])   
        training_output_category.append(row[Analysis_Result_columnName])   
    else:
        testing_description.append(line)
        testing_corpus.append(cleaned)
        testing_ticket_numbers.append(row[Analysis_ticket_columnName])   


label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(training_output_category)

    
#Count Vectoriser then tidf transformer
transformer = TfidfVectorizer(stop_words='english')
tfidf = transformer.fit_transform(training_corpus)
#print(tfidf.shape)                        

# FIRST DO UNSUPERVISED CLUSTERING ON TEXT USING KMeans

modelkmeans = KMeans(n_clusters=num_clusters)
modelkmeans.fit(tfidf)
clusters = modelkmeans.labels_.tolist()

classification_dic={'Issue': training_description, 'Transformed Data':training_corpus, 'Machine Cluster':clusters, 'Human Classification': training_output_category} #Creating dict having doc with the corresponding cluster number.
frame=pd.DataFrame(classification_dic, index=[training_ticket_numbers], columns=['Issue', 'Transformed Data', 'Machine Cluster', 'Human Classification']) # Converting it into a dataframe.

#print("\n")
#print(frame) #Print the doc with the labeled cluster number.
#print("\n")
#print(frame['Machine Classification'].value_counts()) #Print the counts of doc belonging to each cluster.

# save to file
frame.to_excel(input_file + "_KMeans_Clusters.xlsx")
print ("Resuls written to " + input_file + "_KMeans_Clusters.xlsx")

lookup = frame.groupby(['Machine Cluster', 'Human Classification'])['Human Classification'].agg({'no':'count'})
mask = lookup.groupby(level=0).agg('idxmax')
lookup = lookup.loc[mask['no']]
lookup = lookup.reset_index()
lookup = lookup.set_index('Machine Cluster')['Human Classification'].to_dict()


# SECOND do Clustering the document with KNN classifier - Supervised learning
# Only possible when you have Manual classification
modelknn = KNeighborsClassifier(n_neighbors=25)
modelknn.fit(tfidf, integer_encoded)


''' PREDICT THE DATA WITH KNN AND KMEANS '''

print (f"Testing data count {len(testing_corpus)}")
#Count Vectoriser then tidf transformer
testing_tfidf = transformer.transform(testing_corpus)
 

predicted_labels_knn = modelknn.predict(testing_tfidf )
predicted_labels_kmeans = modelkmeans.predict(testing_tfidf )



classification_dic={'Issue': testing_description, 'Transformed Data' : testing_corpus, 'Machine Cluster':predicted_labels_kmeans} #Creating dict having doc with the corresponding cluster number.
predicted_frame=pd.DataFrame(classification_dic, index=[testing_ticket_numbers], columns=['Issue', 'Transformed Data', 'Machine Cluster']) # Converting it into a dataframe.
predicted_frame["KMeans Machine Classification"] = predicted_frame['Machine Cluster'].map(lookup)
predicted_frame["KNN Machine Classification"] = label_encoder.inverse_transform(predicted_labels_knn)

# save to file
predicted_frame.to_excel(input_file + "_Result.xlsx")
print ("Resuls written to " + input_file + "_Result.xlsx")

