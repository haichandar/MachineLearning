# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:28:17 2019

@author: Chandar_S
"""

import sys

if __name__ == "__main__":
    
    if (len(sys.argv) == 1):
        input = "Courses\n\
TRAN~Transfiguration~1~2011-2012~Minerva McGonagall\n\
CHAR~Charms~1~2011-2012~Filius Flitwick\n\
Students\n\
SLY2301~Hannah Abbott\n\
SLY2302~Euan Abercrombie\n\
SLY2303~Stewart Ackerley\n\
SLY2304~Bertram Aubrey\n\
SLY2305~Avery\n\
SLY2306~Malcolm Baddock\n\
SLY2307~Marcus Belby\n\
SLY2308~Katie Bell\n\
SLY2309~Sirius Orion Black\n\
Grades\n\
TRAN~1~2011-2012~SLY2301~AB\n\
TRAN~1~2011-2012~SLY2302~B\n\
TRAN~1~2011-2012~SLY2303~B\n\
TRAN~1~2011-2012~SLY2305~A\n\
TRAN~1~2011-2012~SLY2306~BC\n\
TRAN~1~2011-2012~SLY2308~A\n\
TRAN~1~2011-2012~SLY2309~AB\n\
CHAR~1~2011-2012~SLY2301~A\n\
CHAR~1~2011-2012~SLY2302~BC\n\
CHAR~1~2011-2012~SLY2303~B\n\
CHAR~1~2011-2012~SLY2305~BC\n\
CHAR~1~2011-2012~SLY2306~C\n\
CHAR~1~2011-2012~SLY2307~B\n\
CHAR~1~2011-2012~SLY2308~AB\n\
EndOfInput"
    else:
        input = sys.argv[1]

#%%
Books, Borrowers, Checkouts = False, False, False
(course_list, student_list, grade_list) = ([], [], [])

student_dic = {}

for i, line in enumerate(input.split("\n")):
    if line == "Courses":
        Books = True
        Borrowers, Checkouts = False, False
    elif line == "Students":
        Books, Checkouts = False, False
        Borrowers = True        
    elif line == "Grades":
        Books, Borrowers = False, False
        Checkouts = True
    elif line == "EndOfInput":
        Books, Borrowers, Checkouts = False, False, False
    else:
        if (Books):
            course_list.append(line)
        if (Borrowers):
            student_list.append(line)
            data = line.split("~")
            student_dic[data[0]] = [[0, 0, 0], data[1], data[0]]
        if (Checkouts):
            grade_list.append(line)
#%%

grade_dic = {"A":10,"AB":9,"B":8,"BC":7,"C":6,"CD":5,"D":4}

import numpy as np
for grade_data in grade_list:
    courseno, _, year, rollno, grade = grade_data.split("~")
    
    student_data = student_dic[rollno][0]
    
    
    student_data[0] += grade_dic[grade]
    student_data[1] += 1
    student_data[2] = np.round(student_data[0] /student_data[1], 2)
    
for stud_key in student_dic:
    stud_data = student_dic[stud_key]
    print (f"{stud_data [2]}~{stud_data [1]}~{stud_data [0][2]}")
