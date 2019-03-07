# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 09:39:37 2019

@author: Chandar_S
"""
import sys

if __name__ == "__main__":
    
    if (len(sys.argv) == 1):
        input = "\
Books\n\
APM-001~Advanced Potion-Making\n\
GWG-001~Gadding With Ghouls\n\
APM-002~Advanced Potion-Making\n\
DMT-001~Defensive Magical Theory\n\
DMT-003~Defensive Magical Theory\n\
GWG-002~Gadding With Ghouls\n\
DMT-002~Defensive Magical Theory\n\
Borrowers\n\
SLY2301~Hannah Abbott\n\
SLY2302~Euan Abercrombie\n\
SLY2303~Stewart Ackerley\n\
SLY2304~Bertram Aubrey\n\
SLY2305~Avery\n\
SLY2306~Malcolm Baddock\n\
SLY2307~Marcus Belby\n\
SLY2308~Katie Bell\n\
SLY2309~Sirius Orion Black\n\
Checkouts\n\
SLY2304~DMT-002~2019-03-27\n\
SLY2301~GWG-001~2019-03-27\n\
SLY2308~APM-002~2019-03-14\n\
SLY2303~DMT-001~2019-04-03\n\
SLY2301~GWG-002~2019-04-03\n\
EndOfInput"
    else:
        input = sys.argv[1]

#%%
Books, Borrowers, Checkouts = False, False, False
book_data, Borrowers_data, Checkout_data  = "", "", ""

for i, line in enumerate(input.split("\n")):
    if line == "Books":
        Books = True
        Borrowers, Checkouts = False, False
    elif line == "Borrowers":
        Books, Checkouts = False, False
        Borrowers = True        
    elif line == "Checkouts":
        Books, Borrowers = False, False
        Checkouts = True
    elif line == "EndOfInput":
        Books, Borrowers, Checkouts = False, False, False
    else:
        if (Books):
            book_data += line + '\n'
        if (Borrowers):
            Borrowers_data += line + '\n'
        if (Checkouts):
            Checkout_data += line + '\n'

#%%
from io import StringIO
import pandas as pd      

Book_df = pd.read_csv(StringIO(book_data), sep="~", names = ["Accession Number", "Title"])
Borrowers_df = pd.read_csv(StringIO(Borrowers_data), sep="~", names = ["Username", "Full Name"])
Checkout_df = pd.read_csv(StringIO(Checkout_data), sep="~", names = ["Username", "Accession Number", "Due Date"])

merge1_df = pd.merge(Checkout_df, Borrowers_df, on='Username', how='inner')
merge2_df = pd.merge(merge1_df, Book_df, on='Accession Number', how='inner')
merge2_df = merge2_df.sort_values(by=['Due Date', 'Full Name'])

for line in list(merge2_df["Due Date"] + "~" + merge2_df["Full Name"] + "~" +  merge2_df["Accession Number"] + "~" +  merge2_df["Title"]):
    print (line)
    