#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(threshold=np.nan)

# Le csv en entrée doit être un tableau de la forme
# 0 : (string) nom de l'activité
# 1 : (int) id de l'activité (qui doit être unique)
# 2 : (int) timestamp de la mesure
# 3 : (float) battement cardiaque
# 4 : (float) accélération selon x
# 5 : (float) accélération selon y
# 6 : (float) accélération selon z

# La classe suivante s'occupe de construire les fenêtres de données nécessaires
# à partir des données entrées, sous la forme ci-dessus
# Elles ne doivent pas être nécessairement prétaitées (ie enlever les plages 
# inintéressantes)

window_size = 128
data_size = 7

class Pretreatment:
    
    def __init__(self,csv_name):
        csv = open(csv_name,"r")
        self.extraction(csv)
            # remplit le tableau data_extracted
        
    def extraction(self, file): 
        # csv est le fichier csv brut contenenant les données
        # extraction renvoie le talbeau des données (au format numpy)
        data_csv = csv.reader(file, delimiter=";")
        self.data_extracted = []
        for row in data_csv:
            self.data_extracted += [row]
        print(self.data_extracted)
        ### éventuellement trier le tableau s'il y a des problèmes
        # self.data_extracted.sort()
        self.data_extracted = np.array(self.data_extracted)
    
    def window_making(self):
        # découpe les données en fenêtres de données, le tout respecant
            # chaque point est dans une unique fenêtre
            # tous les points d'une fenêtre se suivent exactement
            # pas de points sans aucune info
        i = 0
        while 
    def cut_id(self):
       # trier par indice, parcourir les tableau et spliter dès qu'on trouver un changement
            
        
pr = Pretreatment("../Data/data_shit1.csv")
print(pr.data_extracted)
    

    
    