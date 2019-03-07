# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:28:17 2019

@author: Chandar_S
"""

import sys

if __name__ == "__main__":
    
    if (len(sys.argv) == 1):
        input = "Books\n\
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
import datetime

#Course, Students, Grades = False, False, False
(books_list, borrowers_list, checkouts_list) = ([], [], [])
books, borrowers, checkouts = False, False, False
books_dic = {}
borrowers_dic = {}
checkouts_dic = {}
final_dic ={}
final_list=[]
for i, line in enumerate(input.split("\n")):
    if line == "Books":
        books = True
        borrowers, checkouts = False, False
    elif line == "Borrowers":
        books, checkouts = False, False
        borrowers = True        
    elif line == "Checkouts":
        books, borrowers = False, False
        checkouts = True
    elif line == "EndOfInput":
        books, borrowers, checkouts = False, False, False
    else:
        if (books):
            books_list.append(line)
            booksdata=line.split('~')
            books_dic[booksdata[0]] = [booksdata[1]]
        if (borrowers):
            borrowers_list.append(line)
            borrowersdata=line.split('~')
            borrowers_dic[borrowersdata[0]] = [borrowersdata[1]]
        if (checkouts):
            checkouts_list.append(line)
            checkoutdata=line.split('~')
            checkouts_dic[checkoutdata[0]]=[checkoutdata[1],checkoutdata[2]]
            
#%%

final_list = []
for checkout_data in checkouts_list:
    Borrowerno, Accession_Number, date = checkout_data.split("~")
    title_name = books_dic[Accession_Number][0]
    borrowers_data = borrowers_dic[Borrowerno][0]
    
    final_list.append((date, borrowers_data, Accession_Number, title_name))

final_list.sort(key = lambda x: (x[0], x[1]))

for data in final_list:
    print (f"{data[0]}~{data[1]}~{data[2]}~{data[3]}")
