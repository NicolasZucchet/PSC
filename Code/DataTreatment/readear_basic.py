#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 14:38:51 2017

@author: nicolas
"""

import csv
import matplotlib.pyplot as plt


i_heart_rate = 6
i_timestamp = 9

# extrait les données de base du csv en un tableau de données
def extractor(file_name):
    file = open(file_name,"r")
    datas_csv = csv.reader(file, delimiter=";")
    datas = []
    i = 0
    for row in datas_csv:
        datas += [row]
        i+=1
    traitment_string(datas)
    map(lambda i : string_to_int(datas,i),[i_heart_rate,i_timestamp])
    return datas

# nettoie les données : les 0 en heart_rate et les plages vides. Corrige également les erreurs temporelles (parfois j+15 sans raison)
def cleaner_heart_rate(datas):
    for i in range(1,len(datas)):
        if int(datas[i][i_heart_rate]) <= 30:
            datas[i][i_heart_rate] = datas[i-1][i_heart_rate]
        if int(datas[i][i_timestamp])-int(datas[i-1][i_timestamp]) >= 500000000:
            datas[i][i_timestamp] = int(datas[i-1][i_timestamp])+5
    
            
    
def base(file_name):
    datas = extractor(file_name)
    datas = datas[:-1]
    cleaner_heart_rate(datas)
    time = []
    heart_rate = []
    for row in datas:
        time = time + [int(row[i_timestamp])]
        heart_rate = heart_rate + [int(row[i_heart_rate])]
    return (time,heart_rate)
      
       
   
# Marche pas
# change le type de la colonne i de string à int
def string_to_int(datas,i):
    for row in datas:
        row[i] = int(row[i])
        print(type(row[i]))

# enlève le "' ... '" des données brutes
def traitment_string(datas):
   for row in datas:
       for i in range(len(row)):
           row[i] = row[i].replace("'","")

       
(time,heart_rate)=base("data_test.csv")
plt.plot(time,heart_rate)
plt.show()