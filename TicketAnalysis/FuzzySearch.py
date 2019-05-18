# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:43:08 2019

@author: Chandar_S
"""

from fuzzywuzzy import fuzz
from fuzzywuzzy import process


master_file = open("Master_List.txt","r")
master_choices = master_file.readlines()

query_file = open("Serach_List.txt","r")
query_list = query_file.readlines()

for query in query_list:
    # Get a list of matches ordered by score, default limit to 5
#    print (process.extract(query, master_choices))
    # If we want only the top one
    print(query.rstrip('\n\r') + ","+ process.extractOne(query, master_choices)[0].rstrip('\n\r') +","+ str(process.extractOne(query, master_choices)[1]))
