# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 17:28:38 2019

@author: Chandar_S
"""

#import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

import numpy as np
import pandas as pd
import string
from PIL import ImageTk, Image
import PIL

from tkinter import *
from tkinter import ttk
from tkinter import filedialog

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import re

class mclass:

    def __init__(self,  window):
        self.window = window
        
        window.title("AI Ops Suite")
#        window.geometry("+50+150")

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
        self.fileName.insert(END, 'T&ADataForAnalysis - Sample.xlsx')
        self.fileName.grid (row=1, column=1)

        self.Sheetlabel = Label( window, text="Sheet Name" )
        self.Sheetlabel.grid (row=2, column=0)
        
        self.sheetName = Entry(window, relief=RIDGE, width=50)
        self.sheetName.insert(END, 'BaseData')
        self.sheetName.grid (row=2, column=1)
#
        self.button = Button (window, text="Read Data", command=self.ReadExcelData)
        self.button.grid(row=3, column=1)
#%%
    def ReadExcelData(self):
        try:
            self.excel_data=pd.read_excel(self.fileName.get(), sheet_name=self.sheetName.get())
                
            column_data = self.excel_data.iloc[0:0,0:30] #Selecting the column that has text.
    
            ticketlabel = Label( window, text="STEP 1: \n Select the column which has unique identifier \n (ex: ticket number)" )
            sourcelabel = Label( window, text="STEP 2: \n Select the ticket data \n to analyze" )
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
            ticket_data[New_Analysis_columnName] = ticket_data[Analysis_primary_columnNames[0]]
            ticket_data[New_Analysis_columnName]
            ticket_data.drop(columns=[Analysis_primary_columnNames[0]])
            Analysis_primary_columnNames.remove(Analysis_primary_columnNames[0])
            for item in Analysis_primary_columnNames:
                ticket_data[New_Analysis_columnName] = ticket_data[New_Analysis_columnName] + " " + ticket_data[item] 
    
            ticket_data.drop(columns=Analysis_primary_columnNames)
    
            for index,row in ticket_data.iterrows():
    #            print (row[New_Analysis_columnName])
                
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
            
            label_encoder = LabelEncoder()
            integer_encoded = label_encoder.fit_transform(training_output_category)
            
            # Perform unsupervised clustering and get cluster resullts
            modelkmeans, clusters = self.PerformClustering(training_corpus);
            
            # IF NO EXISTING MANUAL TAG AVAILABLE, PERFORM UNSUPERVISED LEARNING
            if Analysis_Result_columnName is None:         
    
                # Analyze the clustering and come up with tagging to plot and generate excel
                plot_frame, cluster_themes_dict = self.AnalyzeClustering(clusters, modelkmeans, training_corpus)
    
                def tagCluster(cluster_no):
                    # return the first tagging
                    return cluster_themes_dict[cluster_no][0][0]
                
                classification_dic={'Issue': training_description, 'Transformed Data':training_corpus, 'Machine Cluster':clusters, 'Machine Tag': list(map(tagCluster, clusters))} 
                excel_frame=pd.DataFrame(classification_dic, index=[training_ticket_numbers], columns=['Issue', 'Transformed Data', 'Machine Cluster', 'Machine Tag'])        
    
                # Show your results in pop-up, a pareto chart and a summary of clusters
                self.PlotResults(plot_frame, excel_frame, input_file)
            else:
                
                transformer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(5,6), max_features=5000)
                tfidf = transformer.fit_transform(training_corpus)
                
                classification_dic={'Issue': training_description, 'Transformed Data':training_corpus, 'Machine Cluster':clusters, 'Human Tag': training_output_category} #Creating dict having doc with the corresponding cluster number.
                frame=pd.DataFrame(classification_dic, index=[training_ticket_numbers], columns=['Issue', 'Transformed Data', 'Machine Cluster', 'Human Tag']) # Converting it into a dataframe.
    
                # do prediction only if testing data is available
                if len(testing_corpus) > 0:
    #                lookup = frame.groupby(['Machine Cluster', 'Human Classification'])['Human Classification'].agg({'no':'count'})
    #                mask = lookup.groupby(level=0).agg('idxmax')
    #                lookup = lookup.loc[mask['no']]
    #                lookup = lookup.reset_index()
    #                lookup = lookup.set_index('Machine Cluster')['Human Classification'].to_dict()
        
                    modelknn = KNeighborsClassifier(n_neighbors=25)
                    modelknn.fit(tfidf, integer_encoded)
                    testing_tfidf = transformer.transform(testing_corpus)
                    
                    predicted_labels_knn = modelknn.predict(testing_tfidf )
    #                predicted_labels_kmeans = modelkmeans.predict(testing_tfidf )
                    
                    classification_dic={'Issue': testing_description, 'Transformed Data' : testing_corpus, 'Machine Cluster':predicted_labels_kmeans} #Creating dict having doc with the corresponding cluster number.
                    predicted_frame=pd.DataFrame(classification_dic, index=[testing_ticket_numbers], columns=['Issue', 'Transformed Data', 'Machine Cluster']) # Converting it into a dataframe.
                    predicted_frame["New Tagging"] = label_encoder.inverse_transform(predicted_labels_knn)
                    
                    # save to file
                    predicted_frame.to_excel(input_file + "_Result.xlsx")
                    messagebox.showinfo("Success", "Predicted Results written to " + input_file + "_Result.xlsx")
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
        transformer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', stop_words='english')
#        transformer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
        tfidf = transformer.fit_transform(training_corpus)

        Sum_of_squared_distances = []
        k_step = 10
        k_start = 10
        k_max = 50
        K = range(k_start, k_max, k_step)
        for k in K:
            km = KMeans(n_clusters=k)
            km = km.fit(tfidf)
            Sum_of_squared_distances.append(km.inertia_)
        
        k_optimal = k_start + (Sum_of_squared_distances.index(min(Sum_of_squared_distances)) + 1) * k_step
                
        fig = Figure(figsize=(4,4))
        plt = fig.add_subplot(111)
        plt.plot(K, Sum_of_squared_distances, 'bx-')
        plt.set_title ("Elbow Method For Optimal cluster #", fontsize=12)
        plt.set_ylabel("Sum_of_squared_distances", fontsize=7)
        plt.set_xlabel("k", fontsize=7)

        canvas = FigureCanvasTkAgg(fig, master=self.window)
        canvas.get_tk_widget().grid(row=7, column=1)
        canvas.draw()

        print (f"clusters chosen {k_optimal}")
        # FIRST DO UNSUPERVISED CLUSTERING ON TEXT USING KMeans
        modelkmeans = KMeans(n_clusters=k_optimal)
        modelkmeans.fit(tfidf)
        clusters = modelkmeans.labels_.tolist()
        return modelkmeans, clusters

#%%
    def AnalyzeClustering(self, clusters, modelkmeans, training_corpus):
        # FIND SIGNIFICANT TERMS IN EACH CLUSTER
        
        # USE WORD GRAM FOR FINDING SIGNIFICANT TERM
        xvalid_tfidf_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(5,6), max_features=5000)
        
        cluster_count_list = []
        cluster_tag_list = []
        cluster_themes_dict = {}
        for i in set(clusters):    
            current_cluster_data = [training_corpus[x] for x in np.where(modelkmeans.labels_ == i)[0]]        
            
            try:
                xvalid_tfidf_ngram.fit_transform(current_cluster_data)
                current_tf_idfs = dict(zip(xvalid_tfidf_ngram.get_feature_names(), xvalid_tfidf_ngram.idf_))
                tf_idfs_tuples = current_tf_idfs.items()
                cluster_themes_dict[i] = sorted(tf_idfs_tuples, key = lambda x: x[1])[:1]
                cluster_tag_list.append(str(cluster_themes_dict[i][0][0]))
#                    cluster_tag_list.append("".join(format([x[0] for x in cluster_themes_dict[i]])))
            except:
                cluster_tag_list.append(current_cluster_data[0])
                cluster_themes_dict[i] = (current_cluster_data[0], 0)
        
            cluster_count_list.append(clusters.count(i))
        
        def NameCluster(cluster_no):
            return "" + str(cluster_no+1)
        
        plot_frame = pd.DataFrame({'Cluster Count':cluster_count_list, 'Machine Tag': cluster_tag_list, 'Cluster': list(map(NameCluster, set(clusters)))},
                                index = set(clusters),
                                columns=['Machine Tag', 'Cluster Count', 'Cluster'])        

        return plot_frame, cluster_themes_dict
    #%%
    def PlotResults (self, plot_frame, excel_frame, input_file):
        
#        print (excel_frame)
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
#            canvas.draw()

        tree = ttk.Treeview(popup_window)
        
        tree["columns"]=("one","two")
        tree.column("one", width=500 )
        tree.column("two", width=500)
        tree.heading("one", text="Formatted Input Data")
        tree.heading("two", text="Raw Input Data")
                
        i = 0
        for clusterindex, row in plot_frame.iterrows():
            id2 = tree.insert("", i, row["Cluster"], text=f'Category {clusterindex + 1} - Machine Tag -> {row["Machine Tag"].upper()} - {row["Cluster Count"]} tickets')
            
            for excelindex, excelrow in excel_frame.iterrows():
                if (clusterindex == excelrow['Machine Cluster'] ):
                    tree.insert(id2, "end", excelindex, text=excelindex, values=(excelrow['Transformed Data'],excelrow['Issue']))
            i+= 1
            
        canvas.get_tk_widget().pack()
        tree.pack()

        exportbutton = Button(popup_window, text="Export the Results", command=self.ExportData(excel_frame, input_file))
        exportbutton.pack()

#%%
window= Tk()
start= mclass (window)
window.mainloop()