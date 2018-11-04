import csv
import matplotlib.pyplot as plt
import time
import datetime

time_offset = 3600000
i_heart_rate = 6
heart_rate_useless = 7
i_timestamp = 8
# Apparemment, c'est le deuxième timestamp qui est important qui importe le plus
i_heartrate_time_offset = 2
# Colonne de timestamp pour le pouls offset de 2 colonnes
x_gyroscope = 10
y_gyroscope = 11
z_gyroscope = 12
path_name = ""
#path_name pour des chemins relatifs
nW = 256

# extrait les données de base du csv en un tableau de données
def extractor(file_name):
    file = open(file_name,"r")
    datas_csv = csv.reader(file, delimiter=",")
    datas = []
    i = 0
    for row in datas_csv:
        datas += [row]
        i+=1
    traitement_string(datas)
    datas.pop(0)
    return datas
    
def time_heart_rate(file_name):
    datas = extractor(file_name)
    datas.sort(key = lambda x: x[i_timestamp+i_heartrate_time_offset])
    delete_column(datas, heart_rate_useless)
    string_to_int(datas,i_timestamp+i_heartrate_time_offset)
    string_to_int(datas,i_heart_rate)
    time = []
    heart_rate = []
    timestamp=int(datas[0][i_timestamp+i_heartrate_time_offset])
    for row in datas:
        # nettoie les données : les 0 en heart_rate. Corrige également les erreurs temporelles (parfois j+15 sans raison)
        if int(row[i_heart_rate])>30 and (int(row[i_timestamp+i_heartrate_time_offset])-timestamp <= 500000000) :
            timestamp=int(row[i_timestamp+i_heartrate_time_offset])
            time = time + [timestamp]
            heart_rate = heart_rate + [int(row[i_heart_rate])]
    return (time,heart_rate)

#time_gyroscope fonctionne également pour l'accélération : même format
def time_gyroscope(file_name):
    datas = extractor(file_name)
    datas.sort(key = lambda x: x[i_timestamp])
    string_to_float(datas,x_gyroscope)
    string_to_float(datas,y_gyroscope)
    string_to_float(datas,z_gyroscope)
    time = []
    gyroscope = []
    timestamp=int(datas[0][i_timestamp])
    for row in datas:
        # nettoie les données : corrige les erreurs temporelles (parfois j+15 sans raison)
        if (int(row[i_timestamp])-timestamp <= 500000000) :
            timestamp=int(row[i_timestamp])
            time = time + [timestamp]
            gyroscope = gyroscope + [[row[x_gyroscope],row[y_gyroscope],row[z_gyroscope]]]
    return (time,gyroscope)


# change le type de la colonne i de string à int
def string_to_int(datas,i):
    for row in datas:
        row[i] = int(row[i])

def string_to_float(datas,i):
    for row in datas:
        row[i] = float(row[i])

def delete_column(datas,i):
    for row in datas:
        row.pop(i)

# enlève le "' ... '" des données brutes
def traitement_string(datas):
    for row in range(len(datas)):
        for i in range(len(datas[row])):
            datas[row][i] = datas[row][i].replace("'","")
            datas[row][i] = datas[row][i].replace(" ","")

# convertit le jour et l'heure en timestamp (avec millisecondes) pour l'ordinateur
# attention à l'heure dété et l'heure d'hiver
def date_to_timestamp(date):
    return 1000*int(datetime.datetime.strptime(date,"%Y-%m-%d-%H:%M:%S").timestamp()) + time_offset

def streamline(file_name_heart,file_name_gyro,file_name_linear_acceleration):
    (timeh, datash) = time_heart_rate(file_name_heart)
    (timeg, datasg) = time_gyroscope(file_name_gyro)
    (timela, datasla) = time_gyroscope(file_name_linear_acceleration)
    debut = max(timeh[0], timeg[0], timela[0])
    fin = min(timeh[-1], timeg[-1], timela[-1])
    streamlined = []
    courant_h = 0
    courant_g = 0
    courant_la = 0
    t = debut
    while debut <= t and t <= fin:
        while timeh[courant_h+1] < t or timeg[courant_g+1] < t or timela[courant_la+1] < t:
            courant_h += int(timeh[courant_h+1] < t)
            courant_g += int(timeg[courant_g+1] < t)
            courant_la += int(timela[courant_la+1] < t)
        mean = int((timeg[courant_g] + timela[courant_la])/2)
        t = max(timeg[courant_g+1], timela[courant_la+1])+1
        streamlined.append([mean,datash[courant_h],datasg[courant_g],datasla[courant_la]])
    return streamlined

# À retravailler selon le format du fichier d'activités
'''
def add_activity(activity, datas):
    pointeur = 0
    for act in activity:
        debut = date_to_timestamp(act[1])
        fin = date_to_timestamp(act[2])
        while int(datas[pointeur][i_timestamp]) <= debut or int(datas[pointeur][i_timestamp]) >= fin :
            pointeur += 1
        nombre_courant = 0
        l = []
        while debut <= int(datas[pointeur+nombre_courant][i_timestamp]) and int(datas[pointeur+nombre_courant][i_timestamp]) <= fin:
             l.append(nombre_courant)
             nombre_courant+=1
        for i in l:
            if datas[pointeur+i][-1] != act[0]:
                datas[pointeur+i].append(act[0])
'''

# decoupage_fenetres prend les données traitées via streamline, auquelles on a ajouté les identifiants des activités. decoupage_fenetres cherche des fenêtres de nW données consécutives de même activité et les ressort dans une liste qui a cette forme : matrix[i] correspond à la ième fenêtre trouvé après un passage linéaire dans le tableau ; matrix[i] est une liste avec en premier élémént l'id de l'activité puis les nW données de la ième fenêtre.
def decoupage_fenetres(datas):
    matrix = []
    t = 0
    while t < len(datas) - nW:
        id_act = datas[t][-1]
        boolean = True
        for i in range(nW):
            if boolean and datas[t+i][-1] != id_act:
                boolean = False
                t += i
        if boolean:
            matrix.append([id_act])
            for i in range(nW):
                matrix[-1].append(datas[t+i])
            t += nW
    return matrix

(time,heart_rate)=time_heart_rate(path_name+"test-psc-heart_rate.csv")
streamlined = streamline(path_name+"test-psc-heart_rate.csv", path_name+"test-psc-gyroscope.csv", path_name+"test-psc-linear_acceleration.csv")
plt.plot(time,heart_rate)
plt.show()
