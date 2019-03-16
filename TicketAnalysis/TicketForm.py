# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 17:28:38 2019

@author: Chandar_S
"""

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
import pylab as pl
from pylab import rc

import numpy as np
import pandas as pd
import string
from PIL import ImageTk
import PIL

from tkinter import *
from tkinter import ttk
from tkinter import filedialog

from sklearn import model_selection, metrics, linear_model, naive_bayes, svm, ensemble, decomposition
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
import xgboost
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers


from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import re

class mclass:

    def __init__(self,  window, embeddings_index):
        self.window = window
        self.embeddings_index = embeddings_index
        
        window.title("AI Ops Suite")

        image = PIL.Image.open("Images/MainLogo.png")
        image = image.resize((112, 56), PIL.Image.ANTIALIAS)
        img = ImageTk.PhotoImage(image)
        panel = Label(window, image = img, width=150, height=100, justify="left")
        panel.image = img
        panel.grid(row=0, column=0)

        self.Header= Label( window, text="AI Ops - Ticket Analytics", justify="center" )
        self.Header.config(fg="teal", font=("Helvetica", 30))
        self.Header.grid(row=0, column=1)

        image = PIL.Image.open("Images/AI.png")
        image = image.resize((100, 100), PIL.Image.ANTIALIAS)
        img = ImageTk.PhotoImage(image)
        panelright = Label(window, image = img, width=150, height=100, justify="right")
        panelright.image = img
        panelright.grid(row=0, column=2)

        self.Filelabel = Label( window, text="EXCEL File Name" )
        self.Filelabel.grid(row=1, column=0)

        def browsefunc():
            filename = filedialog.askopenfilename()
            self.fileName.delete(0, END)
            self.fileName.insert(END, filename)

#            self.fileName.config(text=filename)
                
        browsebutton = Button(window, text="Browse", command=browsefunc, justify="left")
        browsebutton.grid(row=1, column=2)

        self.fileName = Entry(window, relief=RIDGE, width=50)
        self.fileName.insert(END, 'T&ADataForAnalysis_NonCluster.xlsx')
        self.fileName.grid (row=1, column=1)

        self.Sheetlabel = Label( window, text="Sheet Name" )
        self.Sheetlabel.grid (row=2, column=0)
        
        self.sheetName = Entry(window, relief=RIDGE, width=50)
        self.sheetName.insert(END, 'Sheet1')
        self.sheetName.grid (row=2, column=1)

        self.button = Button (window, text="Read Data", command=self.ReadExcelData)
        self.button.grid(row=3, column=1)
#%%
    def ReadExcelData(self):
        try:
            self.excel_data=pd.read_excel(self.fileName.get(), sheet_name=self.sheetName.get())
                
            column_data = self.excel_data.iloc[0:0,0:30] #Selecting the column that has text.
    
            ticketlabel = Label( window, text="STEP 1: \n Select the column which has unique identifier \n (ex: ticket number)" )
            sourcelabel = Label( window, text="STEP 2: \n Select the ticket data \n to analyze\n(Multiple fields can be selected)" )
            targetlabel = Label( window, text="STEP 3: \n Select the classification column \n(semi-human classified data) \n to be predicted" )
            
            self.ticket_column_list = Listbox(self.window, selectmode=SINGLE, width=50, height=10)
            self.ticket_column_list.configure(exportselection=False)
            self.source_column_list = Listbox(self.window, selectmode=EXTENDED, width=50, height=10)
            self.source_column_list.configure(exportselection=False)
            self.target_column_list = Listbox(self.window,selectmode=SINGLE, width=50, height=10)
            self.source_column_list.configure(exportselection=False)
            
            self.target_column_list .insert(END, "I DON'T HAVE THIS DATA")
            
            self.source_column_dic = {}
            ind = 0
            for item in column_data:
                self.ticket_column_list.insert(END, item)
                self.source_column_list.insert(END, item)
                self.source_column_dic[ind] = item
                self.target_column_list .insert(END, item)
                ind = ind + 1

            Scrollbar(self.ticket_column_list, orient="vertical")
            Scrollbar(self.source_column_list, orient="vertical")                
            Scrollbar(self.target_column_list, orient="vertical")                
            
            ticketlabel.grid(row=4, column=0)
            self.ticket_column_list.grid(row=5, column=0)
            sourcelabel.grid(row=4, column=1)
            self.source_column_list.grid(row=5, column=1)
            targetlabel.grid(row=4, column=2)
            self.target_column_list .grid(row=5, column=2)
    
            button = Button (window, text="Analyze Tickets", command=self.AnalyzeTickets)
            button.grid(row=6, column=1)
        except Exception as e:
            messagebox.showerror("Read Error", str(e))                
#%%
    def AnalyzeTickets(self):
        try:
            input_file = "Mytest"
            
            items = self.source_column_list.curselection()
            Analysis_primary_columnNames = [self.source_column_dic[int(item)] for item in items]
            Analysis_Result_columnName = None if self.target_column_list.curselection()[0] == 0 else self.target_column_list.get(ACTIVE)
            Analysis_ticket_columnName  = self.ticket_column_list.get(ACTIVE)
    
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
            All_ticket_numbers=[]
            training_corpus=[]
            training_description=[]
            testing_corpus=[]
            testing_description=[]
            training_ticket_numbers=[]
            testing_ticket_numbers=[]
            training_output_category=[]
    
            ticket_data = self.excel_data.iloc[:,0:30] #Selecting the column that has text.
    
    
            # Trying to add a new column which will hold all the selected columns
            New_Analysis_columnName = "MasterIssue"
            ticket_data[New_Analysis_columnName] = ticket_data[Analysis_primary_columnNames[0]].copy()
#            ticket_data.drop(columns=[Analysis_primary_columnNames[0]])
            Analysis_primary_columnNames.remove(Analysis_primary_columnNames[0])
            for item in Analysis_primary_columnNames:
                ticket_data[New_Analysis_columnName] = ticket_data[New_Analysis_columnName] + " " + ticket_data[item]
    
#            ticket_data.drop(columns=Analysis_primary_columnNames)
    
            for index,row in ticket_data.iterrows():
                
                line = ""
                if (row[New_Analysis_columnName] and str(row[New_Analysis_columnName]) != 'nan' ):
                    line = str(row[New_Analysis_columnName])
    
                    line = line.strip()
                    cleaned = clean(line)
                    cleaned = ' '.join(cleaned)
            
                    ''' IF MANUAL CLASSFICATION IS AVAILABLE, PUT THEM INTO TRAINING, ELSE TESTING'''
                    if (Analysis_Result_columnName is None or str(row[Analysis_Result_columnName]) != 'nan'):
                        training_description.append(line)
                        training_corpus.append(cleaned)
                        # Add ticket number for indexing
                        training_ticket_numbers.append(row[Analysis_ticket_columnName])   
                        if not Analysis_Result_columnName is None:
                            training_output_category.append(row[Analysis_Result_columnName])   
                    else:
                        testing_description.append(line)
                        testing_corpus.append(cleaned)
                        testing_ticket_numbers.append(row[Analysis_ticket_columnName])   
                    All_ticket_numbers.append(row[Analysis_ticket_columnName])
                        
            
            # IF NO EXISTING MANUAL TAG AVAILABLE, PERFORM UNSUPERVISED LEARNING
            if Analysis_Result_columnName is None:         
    
                # Perform unsupervised clustering and get cluster resullts
                cluster_labels, clusters = self.PerformClustering(training_corpus);

                # Analyze the clustering and come up with tagging to plot and generate excel
                plot_frame, cluster_themes_dict = self.AnalyzeClustering(clusters, cluster_labels, training_corpus)
    
                def tagCluster(cluster_no):
                    # return the first tagging
                    return cluster_themes_dict[cluster_no]
                
                classification_dic={'Issue': training_description, 'Transformed Data':training_corpus, 'Machine Cluster':clusters, 'Machine Tag': list(map(tagCluster, clusters))} 
                excel_frame=pd.DataFrame(classification_dic, index=[training_ticket_numbers], columns=['Issue', 'Transformed Data', 'Machine Cluster', 'Machine Tag'])        
    
                # Show your results in pop-up, a pareto chart and a summary of clusters
                self.PlotResults(plot_frame, excel_frame, input_file)
            else:
                
                classification_dic={'Issue': training_description, 'Transformed Data':training_corpus, 'Human Tag': training_output_category, 'Machine Tag': training_output_category} #Creating dict having doc with the corresponding cluster number.
                excel_frame=pd.DataFrame(classification_dic, index=[training_ticket_numbers], columns=['Issue', 'Transformed Data', 'Machine Cluster', 'Human Tag', 'Machine Tag']) # Converting it into a dataframe.
    
                # do prediction only if testing data is available
                if len(testing_corpus) > 0:
                    predicted_labels = self.RunTrainingModels(training_corpus, training_output_category, testing_corpus)
                                        
                    classification_dic={'Issue': testing_description, 'Transformed Data' : testing_corpus, 'Machine Tag':predicted_labels} #Creating dict having doc with the corresponding cluster number.
                    predicted_frame=pd.DataFrame(classification_dic, index=[testing_ticket_numbers], columns=['Issue', 'Transformed Data', 'Machine Tag']) # Converting it into a dataframe.                    
                    
                    excel_frame = pd.concat([excel_frame, predicted_frame], sort=False)
                    

                    plot_frame = excel_frame[['Machine Tag','Issue']].groupby('Machine Tag').count()
                    plot_frame.columns = ["Cluster Count"]
                    plot_frame["Cluster"] = plot_frame.index
                    plot_frame["Machine Tag"] = plot_frame.index
                    
                    self.PlotResults(plot_frame, excel_frame, input_file)
                    
        except Exception as e:
            messagebox.showerror("Processing Error", "An unexpected error occurred. Please check if you have selected all 3 input data \n Error: " + str(e))                

#%%
    def ExportData(self, excel_frame, input_file):
        # save to file
        excel_frame.to_excel(input_file + "_Clusters.xlsx")
        messagebox.showinfo("Success", "Automated Machine Clustering generated and written to " + input_file + "_Clusters.xlsx")

#%%
    def PerformClustering(self, training_corpus):        

        
        #Count Vectoriser then tidf transformer
        transformer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', stop_words=stopwords.words("english"))
#        transformer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
        tfidf = transformer.fit_transform(training_corpus)

        Sum_of_squared_distances = []
        k_step = 1
        k_start = 1
        k_max = 20
        K = range(k_start, k_max + 1, k_step)
        for k in K:
            km = KMeans(n_clusters=k)
            km = km.fit(tfidf)
            Sum_of_squared_distances.append(km.inertia_)
        
        k_optimal = k_start + Sum_of_squared_distances.index(min(Sum_of_squared_distances)) * k_step
                
#        fig = Figure(figsize=(6,4))
#        plt = fig.add_subplot(111)
#        plt.plot(K, Sum_of_squared_distances, 'bx-')
#        plt.set_title ("Elbow Method For Optimal cluster #", fontsize=12)
#        plt.set_ylabel("Sum_of_squared_distances", fontsize=7)
#        plt.set_xlabel("k", fontsize=7)
#
#        canvas = FigureCanvasTkAgg(fig, master=self.window)
#        canvas.get_tk_widget().grid(row=7, column=0)
#        canvas.draw()

        print ("clusters chosen "+ str(k_optimal))
        # FIRST DO UNSUPERVISED CLUSTERING ON TEXT USING KMeans
        modelkmeans = KMeans(n_clusters=k_optimal, init='k-means++', precompute_distances=True)
        modelkmeans.fit(tfidf)
        cluster_labels = modelkmeans.labels_
        clusters = cluster_labels.tolist()
                
        svd = TruncatedSVD(n_components=2, n_iter=7, random_state=42)
        svd.fit(tfidf)
        svd_2d = svd.transform(tfidf)
        
        rc('figure', figsize=(15, 4))
        fig2 = pl.figure('K-means with ' + str(k_optimal) + ' clusters',)
        fig2.clf()
        pl.scatter(svd_2d[:, 0], svd_2d[:, 1], c=cluster_labels, alpha=0.2, cmap='viridis')
        pl.title("Ticket similarities in 2D view. Each circle is a ticket")
        
        canvas = FigureCanvasTkAgg(fig2, master=self.window)
#        canvas.get_tk_widget().grid(row=7, column=2)
        canvas.get_tk_widget().grid(row=7, column=0, columnspan = 3)

        canvas.draw()
        
        return cluster_labels, clusters

#%%
    def AnalyzeClustering(self, clusters, cluster_labels, training_corpus):
        # FIND SIGNIFICANT TERMS IN EACH CLUSTER
                
        cluster_count_list = []
        cluster_tag_list = []
        cluster_themes_dict = {}
        for i in set(clusters):    
            cluster_count_list.append(clusters.count(i))
            current_cluster_data = [training_corpus[x] for x in np.where(cluster_labels == i)[0]]        

            try:
                # USE WORD GRAM FOR FINDING SIGNIFICANT TERM
                xvalid_tfidf_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(4,5), max_features=5000)
                xvalid_tfidf = xvalid_tfidf_ngram.fit_transform(current_cluster_data)
            except:
                # USE WORD GRAM FOR FINDING SIGNIFICANT TERM
                xvalid_tfidf_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,5), max_features=5000)
                xvalid_tfidf = xvalid_tfidf_ngram.fit_transform(current_cluster_data)
                
            try:
                
                components = 3 if xvalid_tfidf.shape[0] > 3 else xvalid_tfidf.shape[0]
                NMF_model = decomposition.NMF(n_components=components, 
                                random_state=1,
                                beta_loss='kullback-leibler',
                                solver='mu', max_iter=1000,
                                alpha=.1, l1_ratio=.5)           
                X_topics = NMF_model .fit_transform(xvalid_tfidf)
                topic_word = NMF_model .components_ 
                vocab = xvalid_tfidf_ngram.get_feature_names()
                
                n_top_words = 1
                topic_list = []
                for _, topic_dist in enumerate(topic_word):
                    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
                    topic_list.append(' '.join(topic_words).upper())
                cluster_themes_dict[i] = ' ||| '.join(topic_list)
                cluster_tag_list.append(cluster_themes_dict[i])
            except Exception as ex:
                print ("Exception", ex)

        def NameCluster(cluster_no):
            return "" + str(cluster_no+1)
        
        plot_frame = pd.DataFrame({'Cluster Count':cluster_count_list, 'Machine Tag': cluster_tag_list, 'Cluster': list(map(NameCluster, set(clusters)))},
                                index = set(clusters),
                                columns=['Machine Tag', 'Cluster Count', 'Cluster'])        

        return plot_frame, cluster_themes_dict
    #%%
    def PlotResults (self, plot_frame, excel_frame, input_file):
        
        popup_window = Toplevel(self.window)

        popup_window.title("Results")
        plot_frame = plot_frame.sort_values(by='Cluster Count',ascending=False)
        plot_frame["Cumulative Percentage"] = plot_frame['Cluster Count'].cumsum()/plot_frame['Cluster Count'].sum()*100
        
#        print (f'{plot_frame["Cluster Count"]}  {plot_frame["Cumulative Percentage"]}')
        fig = Figure(figsize=(40,5))
        ax = fig.add_subplot(111)
        ax.bar(plot_frame['Cluster'], plot_frame["Cluster Count"], color="C0")
        ax2 = ax.twinx()
        ax2.plot(plot_frame['Cluster'], plot_frame["Cumulative Percentage"], color="C1", marker="D", ms=7)
        ax2.yaxis.set_major_formatter(PercentFormatter())
        
        ax.tick_params(axis="y", colors="C0")
        ax2.tick_params(axis="y", colors="C1")

        fig.legend()
        ax.set_title ("RESULTS - Summary of categorized Ticket Data (Most contributing to least contributing categories)", fontsize=20)
        ax.set_ylabel("# of Tickets", fontsize=10)
        ax.set_xlabel("Category #", fontsize=10)

        canvas = FigureCanvasTkAgg(fig, master=popup_window )

        tree = ttk.Treeview(popup_window)
        
        tree["columns"]=("one","two")
        tree.column("one", width=500 )
        tree.column("two", width=500)
        tree.heading("one", text="Formatted Input Data")
        tree.heading("two", text="Raw Input Data")
                
        i = 0
        for clusterindex, row in plot_frame.iterrows():
            try:
                id2 = tree.insert("", i, row["Cluster"], text="Category "+ str(clusterindex + 1) + " -> " + str(row["Cluster Count"]) + " tickets"  +" - " + row["Machine Tag"].upper())            
                for excelindex, excelrow in excel_frame.iterrows():
                    if (clusterindex == excelrow['Machine Cluster'] ):
                        tree.insert(id2, "end", excelindex, text=excelindex, values=(excelrow['Transformed Data'],excelrow['Issue']))
            # handles parameters in different way 
            except:
                id2 = tree.insert("", i, row["Cluster"], text="Tag -> " + row["Machine Tag"].upper() +" - " + str(row["Cluster Count"]) + " tickets")            
                for excelindex, excelrow in excel_frame.iterrows():
                    if (clusterindex == excelrow['Machine Tag'] ):
                        tree.insert(id2, "end", excelindex, text=excelindex, values=(excelrow['Transformed Data'],excelrow['Issue']))

            i+= 1
            
        canvas.get_tk_widget().pack()
        tree.pack()

        exportbutton = Button(popup_window, text="Export the Results", command=self.ExportData(excel_frame, input_file))
        exportbutton.pack()

#%%
    def RunTrainingModels(self, training_corpus, training_output_category, testing_corpus):

        label_encoder = LabelEncoder()
        training_label_encoded = label_encoder.fit_transform(training_output_category)

        # split the dataset into training and validation datasets 
        train_x, valid_x, train_y, valid_y = model_selection.train_test_split(training_corpus, training_label_encoded)

        def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
            if is_neural_net:
                # fit the training dataset on the classifier
                classifier.fit(feature_vector_train, label, batch_size = 1000, epochs=10, validation_split=0.01)

                # predict the labels on validation dataset
                if feature_vector_valid is None:
                    return (-1, classifier)
                else:
                    predictions = classifier.predict(feature_vector_valid)
                    predictions = predictions.argmax(axis=-1)
                    print ("prediction", predictions)

            else:
                # fit the training dataset on the classifier
                classifier.fit(feature_vector_train, label)
                # predict the labels on validation dataset
                predictions = classifier.predict(feature_vector_valid)
            
            return int(metrics.accuracy_score(predictions, valid_y) *100), classifier


        accuracy_list = []
        algorithm_list = []
        classifer_list = []

        # word level tf-idf        
        transformer_new = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=50000)
        tfidf_fit = transformer_new.fit(training_corpus)
        xtrain_tfidf = tfidf_fit.transform(train_x)
        xvalid_tfidf = tfidf_fit.transform(valid_x)

        # ngram level tf-idf 
        tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
        tfidf_vect_ngram.fit(training_corpus)
        xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
        xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

        # characters level tf-idf
        tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
        tfidf_vect_ngram_chars.fit(training_corpus)
        xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
        xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)
  
#        '''
        accuracy, classifier = train_model(KNeighborsClassifier(n_neighbors=25), xtrain_tfidf, train_y, xvalid_tfidf)
        accuracy_list.append(accuracy)
        algorithm_list.append("KNN WL TF-IDF")
        classifer_list.append(classifier)
              
        
        # Linear Classifier on Word Level TF IDF Vectors
        accuracy, classifier  = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
        accuracy_list.append(accuracy)
        algorithm_list.append("LR WL TF-IDF")
        classifer_list.append(classifier)

        # Linear Classifier on Ngram Level TF IDF Vectors
        accuracy, classifier = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
        accuracy_list.append(accuracy)
        algorithm_list.append("LR N-Gram")
        classifer_list.append(classifier)
        
        # SVM on Ngram Level TF IDF Vectors
        accuracy, classifier  = train_model(svm.SVC(), xtrain_tfidf, train_y, xvalid_tfidf)
        accuracy_list.append(accuracy)
        algorithm_list.append("SVM WL TF-IDF")
        classifer_list.append(classifier)
#        '''
        
        '''
        # Naive Bayes on Word Level TF IDF Vectors
        accuracy, classifier  = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
        accuracy_list.append(accuracy)
        algorithm_list.append("NB WL TF-IDF")
        classifer_list.append(classifier)

        # Naive Bayes on Ngram Level TF IDF Vectors
        accuracy, classifier = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
        accuracy_list.append(accuracy)
        algorithm_list.append("NB N-Gram")
        classifer_list.append(classifier)
        # RF on Word Level TF IDF Vectors
 
        accuracy, classifier  = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)
        accuracy_list.append(accuracy)
        algorithm_list.append("RF WL TF-IDF")
        classifer_list.append(classifier)

        # Extereme Gradient Boosting on Word Level TF IDF Vectors
        accuracy, classifier   = train_model(xgboost.XGBClassifier(), xtrain_tfidf.tocsc(), train_y, xvalid_tfidf.tocsc())
        accuracy_list.append(accuracy)
        algorithm_list.append("Xgb WL TF-IDF")
        classifer_list.append(classifier)
        
        # Extereme Gradient Boosting on Character Level TF IDF Vectors
        accuracy, classifier = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram_chars.tocsc(), train_y, xvalid_tfidf_ngram_chars.tocsc())
        accuracy_list.append(accuracy)
        algorithm_list.append("Xgb CL")
        classifer_list.append(classifier)
        '''
        
#        '''
        
        # Reset the training and validation. For neural net, we need more data and will use inbuilt validation in keras
#        train_x, train_y, valid_seq_x, valid_y,  = training_corpus, training_label_encoded, None, None

        # load the pre-trained word-embedding vectors
        if self.embeddings_index is None:
            self.embeddings_index = {}
            for i, line in enumerate(open('glove.6B/glove.6B.300d.txt', encoding="utf8")):
                values = line.split()
                self.embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')

        # create a tokenizer 
        token = text.Tokenizer()
        token.fit_on_texts(training_corpus)
        word_index = token.word_index
        
        # convert text to sequence of tokens and pad them to ensure equal length vectors 
        train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
        valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)
        
        # create token-embedding mapping
        embedding_matrix = np.zeros((len(word_index) + 1, 300))
        for word, i in word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        
        def create_rcnn():
            # Add an Input Layer
            input_layer = layers.Input((70, ))
        
            # Add the word embedding Layer
            embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
            embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)
            
            # Add the recurrent layer
            layers.Bidirectional(layers.GRU(50, return_sequences=True))(embedding_layer)
            
            # Add the convolutional Layer
            conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)
        
            # Add the pooling Layer
            pooling_layer = layers.GlobalMaxPool1D()(conv_layer)
        
            # Add the output Layers
            output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
            output_layer1 = layers.Dropout(0.25)(output_layer1)
            output_layer2 = layers.Dense(units=max(training_label_encoded) + 1, activation="softmax", name="ouput_layer")(output_layer1)
        
            # Compile the model
            model = models.Model(inputs=input_layer, outputs=output_layer2)
            model.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=["sparse_categorical_accuracy"])
            
            return model
        
        classifier = create_rcnn()
        accuracy, _ = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
        accuracy_list.append(accuracy)
        algorithm_list.append("Deep - RCNN")
        classifer_list.append(classifier)

        fig = Figure(figsize=(15,4))
        ax = fig.add_subplot(111)
        ax.bar(algorithm_list, accuracy_list, color="C0")
        
        ax.tick_params(axis="y", colors="C0")

#        fig.legend()
        ax.set_title ("Evaluvated ML Algorithms", fontsize=20)
        ax.set_ylabel("Accuracy %", fontsize=10)
        ax.set_xlabel("ML Algorithm", fontsize=10)

        canvas = FigureCanvasTkAgg(fig, master=self.window )
        canvas.get_tk_widget().grid(row=7, column=0, columnspan = 3)


        selected_algorithm = None
        selected_algorithm_Name = None
        selected_accuracy = -1
        for algorithm_name, algorithm, accuracy in zip(algorithm_list, classifer_list, accuracy_list):
            print ("Algorithm -> " , algorithm_name , " ||| Accuracy -> " , int(accuracy) , "%\n")
            if (selected_accuracy <= accuracy):
                selected_algorithm = algorithm
                selected_algorithm_Name = algorithm_name
                selected_accuracy = accuracy
        
        print ("Selected Algorithm ", selected_algorithm_Name, " Accuracy ", selected_accuracy , "%")

        # decide the transformer based on algorithm
        if "WL TF-IDF" in selected_algorithm_Name:
            testing_data = transformer_new.transform(testing_corpus)
        elif "N-Gram" in selected_algorithm_Name:
           testing_data = tfidf_vect_ngram.transform(testing_corpus)
        elif "CL" in selected_algorithm_Name:
            testing_data = tfidf_vect_ngram_chars.transform(testing_corpus)
        elif "Deep" in selected_algorithm_Name:
           testing_data = sequence.pad_sequences(token.texts_to_sequences(testing_corpus), maxlen=70)
        
        predicted_labels = selected_algorithm.predict(testing_data)
        
        # do a reverse transformation for deep learning outputs
        predicted_labels = label_encoder.inverse_transform(predicted_labels.argmax(axis=-1)) if "Deep" in selected_algorithm_Name else label_encoder.inverse_transform(predicted_labels)
        
#        '''      
        return predicted_labels

#%%
from tqdm import tqdm
window= Tk()
# load the pre-trained word-embedding vectors 
embeddings_index = {}
for i, line in tqdm(enumerate(open('glove.6B/glove.6B.300d.txt', encoding="utf8"))):
    values = line.split()
    embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')

start= mclass (window, embeddings_index)
window.mainloop()