###############################################################################
##                                                                           ##
##                                 READER                                    ##
##                                                                           ##
###############################################################################

#Le Reader permet de transformer simplement le dossier dézzipé
#de He@lsy en une base de données exploitable pour le machine-
#learning (pour toi le Zucc' <3).

#Il suffit d'indiquer ci-dessous le chemin absolu vers le dossier et le fichier
# d'activités (comme dans l'exemple...)

path_name_folder = "/Users/arthurloison/Downloads/data-psc_5a4e2b2834b10/psc/"
path_name_activities = "/Users/arthurloison/Downloads/activities.txt"

#Il te reste à choisir la taille de la fenêtre  que tu souhaite
#Attention si tu mets trop grand, tu auras moins d'activités
#(Il faut choisir une puissance de 2 (2, 4, 8, 16, ...))

taille_fenetre=16

#Enfin après avoir fait Run, tape "main()" dans la console
#C'est fini, tout est dans le dossier data-psc_XXXXXXXXXX/psc/data !!!

#P.S. : Si tu le relances, pense a supprimer le dossier data créé
#sinon ça réécrit par dessus

#N'hésite pas à partager

#Merci à Ayman <3

##============================= Début du code ===============================##
import csv
import datetime
import os

##========================== Constantes du code =============================##
time_offset = 3600000       #Heure d'hiver  
#Raccourcis de colonnes                        
i_heart_rate = 6                                
i_timestamp = 8             #Choix du timestamp             
i_heartrate_time_offset = 3 #Choix du timestamp 
x_gyroscope = 10                                
y_gyroscope = 11                                
z_gyroscope = 12                                

##================================ main =====================================##
#3 étapes: 
#   création d'un unique et beau csv
#   création du tableau réunissant toutes les informations utiles
#   découpage en fenêtres et création des fichiers

def main():
    main_reader(path_name_folder)
    data = streamline(path_name_folder+"psc-heart_rate-final.csv",
                      path_name_folder+"psc-gyroscope-final.csv",
                      path_name_folder+"psc-linear_acceleration-final.csv")
    découpage(data)
    
##============================ main_reader ==================================##
#Recolle les différents csv entre eux

def main_reader(path_name):
    #Suppression fichiers uniques si déjà créés
    if os.path.isfile(path_name+"psc-heart_rate-final.csv"):
        os.remove(path_name+"psc-heart_rate-final.csv")
    if os.path.isfile(path_name+"psc-gyroscope-final.csv"):
        os.remove(path_name+"psc-gyroscope-final.csv")
    if os.path.isfile(path_name+"psc-linear_acceleration-final.csv"):
        os.remove(path_name+"psc-linear_acceleration-final.csv")
    
    #Création fichiers uniques
    heart_rate_file = open(path_name+"psc-heart_rate-final.csv", "a")
    gyroscope_file = open(path_name+"psc-gyroscope-final.csv", "a")
    linear_acceleration_file = open(path_name+
                                    "psc-linear_acceleration-final.csv", "a")
    
    #Ajout premier fichier (avec header)
    for line in open(path_name+"psc-heart_rate.csv"):
        heart_rate_file.write(line)       
    for line in open(path_name+"psc-gyroscope.csv"):
        gyroscope_file.write(line)     
    for line in open(path_name+"psc-linear_acceleration.csv"):
        linear_acceleration_file.write(line)
    
    #Ajout des fichiers suivants
    for num in range(1, 20):
        if os.path.isfile(path_name+"psc-heart_rate"+str(num)+".csv"):
            h = open(path_name+"psc-heart_rate"+str(num)+".csv", "r")
            for line in h:
                heart_rate_file.write(line)          
        if os.path.isfile(path_name+"psc-gyroscope"+str(num)+".csv"):
            g = open(path_name+"psc-gyroscope"+str(num)+".csv", "r")
            for line in g:
                gyroscope_file.write(line)    
        if os.path.isfile(path_name+"psc-linear_acceleration"+str(num)+".csv"):
            la = open(path_name+"psc-linear_acceleration"+str(num)+".csv", "r")
            for line in la:
                linear_acceleration_file.write(line)

##============================== extractor ==================================##
# Extrait les données de base du csv en un tableau de données brutes (sans le 
# header) de la forme : ['Nom', "user_id", ...] (idem csv)

def extractor(file_name):
    file = open(file_name,"r")
    datas_csv = csv.reader(file, delimiter=",")
    datas = []
    for row in datas_csv:
        datas.append(row)
    traitement_string(datas)
    i=0
    #Enlève les lignes de header
    while i<len(datas):
        if datas[i][0]=='Username':
            datas.pop(i)
            i-=1
        i+=1
        
    return datas

##========================= time_heart_rate =================================##
# Extrait les informations utiles du csv et donne un couple de tableaux de la 
# forme : ([timestp1, timestp2, ...],[bpm1, bpm2, ...])

def time_heart_rate(file_name):
    #Extraction des données brutes
    datas = extractor(file_name)
    
    #Tri par date des données                                 
    datas.sort(key = lambda x: x[i_timestamp+i_heartrate_time_offset])  
    
    string_to_int(datas,i_timestamp+i_heartrate_time_offset)
    string_to_int(datas,i_heart_rate)
    
    #Initialisation des tableaux
    time = []                                                           
    heart_rate = []
    
    #Nettoyage des données et implémentation des tableaux
    for row in datas:
        if row[i_heart_rate]>30 :                                       
            time.append(row[i_timestamp+i_heartrate_time_offset])
            heart_rate.append(row[i_heart_rate])
            
    return (time,heart_rate)

##========================= time_gyroscope ==================================##
# Extrait les informations utiles du csv gyroscope ET acceleration et donne un 
# couple de tableaux de la forme :
# ([timestp1, timestp2, ...],[[gyrx1,gyry1,gyrz1],[gyrx2,gyry2,gyrz2], ...])
# ([timestp1, timestp2, ...],[[accx1,accy1,accz1],[accx2,accy2,accz2], ...])


def time_gyroscope(file_name):
    #Extraction des données brutes
    datas = extractor(file_name)  
    
    #Tri par date des données   
    datas.sort(key = lambda x: x[i_timestamp])                
    
    string_to_int(datas,i_timestamp)
    string_to_float(datas,x_gyroscope)
    string_to_float(datas,y_gyroscope)
    string_to_float(datas,z_gyroscope)
    
    #Initialisation des tableaux
    time = []                                                           
    gyroscope = []
    
    #Implémentation des tableaux
    for row in datas:
        time.append(row[i_timestamp])
        gyroscope.append([row[x_gyroscope],row[y_gyroscope],row[z_gyroscope]])
        
    return (time,gyroscope)


##================ string_to_int, string_to_float ===========================##
# Fonctions auxiliaires de conversions d'une colonne i du tableau csv

def string_to_int(datas,i):
    for row in datas:
        row[i] = int(row[i])

def string_to_float(datas,i):
    for row in datas:
        row[i] = float(row[i])

##======================== traitement_string ================================##
#Enlève les guillemets " ' " et " " ".

def traitement_string(datas):
    for row in range(len(datas)):
        for i in range(len(datas[row])):
            datas[row][i] = datas[row][i].replace("'","")
            datas[row][i] = datas[row][i].replace(" ","")
            
##=========================== date_to_timestamp =============================##
#Convertit le jour et l'heure en timestamp (avec milisec) pour l'ordinateur
#Attention à l'heure dété et l'heure d'hiver

def date_to_timestamp(date):
    return 1000*int(datetime.datetime.strptime(date,
                    "%Y-%m-%d-%H:%M:%S").timestamp()) + time_offset

##=========================== streamline ====================================##
#Fonction importante qui assemble toutes les données en un seul tableau

def streamline(file_name_heart,file_name_gyro,file_name_linear_acceleration):
    #Récupération des données de chaque fichier
    (timeh, datash) = time_heart_rate(file_name_heart)                  
    (timeg, datasg) = time_gyroscope(file_name_gyro)
    (timela, datasla) = time_gyroscope(file_name_linear_acceleration)

    #Fenêtre temporelle d'étude
    debut = max(timeh[0], timeg[0], timela[0])
    fin = min(timeh[-1], timeg[-1], timela[-1])
    
    #Init tableau de sortie et indices de parcours des données et du temps
    streamlined = []
    cursorh = 0
    cursorg = 0
    cursorla = 0
    t = debut
    while debut <= t and t <= fin:
        #Récupération des deux valeurs suivantes d'accélération et de gyroscope
        #Màj du heart_rate s'il change entre ces mesures et les précédentes
        while timeh[cursorh+1]<t or timeg[cursorg+1]<t or timela[cursorla+1]<t:
            #ATTENTION si le heart_rate est trop ancien (trous de données)
            #la valeur n'est pas mise à jour et est décorrélée (à vérifier)
            cursorh += int(timeh[cursorh+1] < t)
            cursorg += int(timeg[cursorg+1] < t)
            cursorla += int(timela[cursorla+1] < t)
            
        #Moyenne des deux mesures pour avoir une seule valeur de temps
        mean = int((timeg[cursorg] + timela[cursorla])/2)
        
        #Incrémentation du temps en prenant le temps max+1 des prises de mesure
        #de l'accélération et du gyroscope
        t = max(timeg[cursorg+1], timela[cursorla+1])+1
        #Ajout d'une ligne au tableau
        streamlined.append([mean,datash[cursorh],datasg[cursorg],
                            datasla[cursorla]])
    #Ajout des infos sur les activités
    add_activity(path_name_activities,streamlined)
    
    return streamlined

##=========================== read_activities ===============================##
#Lit le fichier d'activité et renvoie un tableau [id, date1, date2, acti, pos]

def read_activities(filename):
    #Lecture du fichier
    fic = open(filename, "r")
    lines = fic.read().splitlines()
    
    #Init tableau de sortie
    activities=[]
    date=""
    
    #Compteur d'activité
    i=0
    
    for line in lines:
        lgth = len(line)
        #Ligne annoncant la date
        if lgth==8:
            date = line.split(" ")
        else:
            if lgth>1:
                splt=line.split(" ")
                #Ajout d'une activité
                activities.append([i, "20"+date[2]+"-"+date[1]+"-"+date[0]+"-"
                                   +splt[0][0:2]+":"+splt[0][2:4]+":"+"00",
                                      "20"+date[2]+"-"+date[1]+"-"+date[0]+"-"
                                   +splt[1][0:2]+":"+splt[1][2:4]+":"+"00",
                                      splt[2],splt[3]])
            i+=1
            
    return activities

##============================= add_activity ================================##
# Ajoute les activités aux autres données et supprime les valeurs non appairées

def add_activity(filename, data):
    activities=read_activities(filename)
    
    i=0
    while i<len(data):
        findAct = False
        for act in activities:
            
            #Si l'on trouve une activité correspondante
            date1 = date_to_timestamp(act[1])
            date2 = date_to_timestamp(act[2])
            if data[i][0] >=  date1 and data[i][0] <= date2:
                data[i].append(act[0])
                data[i].append(act[3])
                data[i].append(act[4])
                findAct=True
                
        #Si l'on a pas trouvé d'activité, on supprime la ligne
        if(not(findAct)):
            data.pop(i)
            i-=1
        i+=1

##============================= découpage ===================================##
#Découpage de fenêtre par activité et création de fichiers (peu importe si les
#valeures soient consécutives tant que c'est la même id_activity)

def découpage(data):
    #Création du dossier data si besoin
    if not os.path.exists(path_name_folder+"data/"):
        os.makedirs(path_name_folder+"data/")
    
    #Initialisation du découpage
    passed_id=[]
    current_id=0
    current_window=[]
    
    while data != []:
        #Suppression de l'activité précédente
        while  data != [] and (data[0][4] in passed_id):
            data.pop(0)
        if data != []:
            #Début de la nouvelle activité
            current_id=data[0][4]             
            for line in data:
                #Début de la nouvelle fenêtre
                if line[4]==current_id:
                    current_window.append(line)
                    #Si l'on a une fenêtre complète
                    if len(current_window)==16:
                        writeWindow(current_window)
                        current_window=[]
            #Fenêtre incomplète effacée
            passed_id.append(current_id)
            current_window=[]

##============================ writeWindow ==================================##
#Ecriture de la fenêtre

def writeWindow(window):
    #Création des nouveaux dossiers
    if not os.path.exists(path_name_folder+"data/input/"):
        os.makedirs(path_name_folder+"data/input/")
    if not os.path.exists(path_name_folder+"data/output/"):
        os.makedirs(path_name_folder+"data/output/")
    
    #Lecture du fichier activity_labels.txt
    if os.path.isfile(path_name_folder+"data/activity_labels.txt"):
        activity_labels= open(path_name_folder+"data/activity_labels.txt", "r")
        ficactlab = activity_labels.readlines()
        activity_labels.close()
        ficactlab = [x.strip() for x in ficactlab] 
        ficactlab = [x.split(" ") for x in ficactlab]
    else:
        ficactlab=[]
    
    #Fichier output.txt
    output = open(path_name_folder+"data/output/output.txt", "a")
    
    #Màj des fichiers activity_labels.txt et output.txt
    isNewAct=True
    for act in ficactlab:
        if act[1] == window[0][5]+window[0][6]:
            output.write(act[0]+"\n")
            isNewAct=False
            
    #Dans le cas d'une nouvelle activité
    if(isNewAct):
        activity_labels= open(path_name_folder+"data/activity_labels.txt", "a")
        activity_labels.write(str(len(ficactlab)+1)+" "+
                                      window[0][5]+window[0][6]+"\n")
        output.write(str(len(ficactlab)+1)+"\n")
        activity_labels.close()
        output.close()
    
    #Fichiers accx.txt accy.txt accz.txt bpm.txt gyrx.txt gyry.txt gyrz.txt
    bpm = open(path_name_folder+"data/input/bpm.txt", "a")
    
    accx = open(path_name_folder+"data/input/accx.txt", "a")
    accy = open(path_name_folder+"data/input/accy.txt", "a")
    accz = open(path_name_folder+"data/input/accz.txt", "a")
    
    gyrx = open(path_name_folder+"data/input/gyrx.txt", "a")
    gyry = open(path_name_folder+"data/input/gyry.txt", "a")
    gyrz = open(path_name_folder+"data/input/gyrz.txt", "a")
    
    #Ecriture des données de la fenêtre
    for line in window:
        bpm.write(str(line[1])+" ")
        gyrx.write(str(line[2][0])+" ")
        gyry.write(str(line[2][1])+" ")
        gyrz.write(str(line[2][2])+" ")
        accx.write(str(line[3][0])+" ")
        accy.write(str(line[3][1])+" ")
        accz.write(str(line[3][2])+" ")
        
    #Retour à la ligne en fin de fenêtre
    bpm.write("\n")
    gyrx.write("\n")
    gyry.write("\n")
    gyrz.write("\n")
    accx.write("\n")
    accy.write("\n")
    accz.write("\n")
    
    #Fermeture des fichiers
    bpm.close()
    gyrx.close()
    gyry.close()
    gyrz.close()
    accx.close()
    accy.close()
    accy.close()