# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import time
from visualisation import matrice_confusion as mc

#permet d'avoir l'heure à laquelle le code commence pour savoir sa durée
localtime = time.asctime( time.localtime(time.time()) )
print ("\nHeure début:",localtime)

#lecture des données 
data = pd.read_excel('creditcard.xlsx');
# print (data);


#Séparation des données, nous recherchons Y = class pour savoir si c'est une fraude
X = data[["Time", "V1", "V2", "V3", "V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"]];
Y = data["Class"];
# print("x",X) # verification qu'on a les bonnes données
# print("y",Y)

#Vérification qu'on a bien "492 fraudes sur les 284 807 transactions" (informations données dans le contenu)
transac_frauduleuse=len(data[Y==1])
#print("transaction frauduleuse",transac_frauduleuse)

# Nous allons maintenant entrainer le code 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#utilisons de la fonction random forest 
foret=RandomForestClassifier(n_estimators=50) # la valeur par défaut= 100 pour le nombre d'arbre dans la forêt

foret.fit(X_train,Y_train)

#Nous prédisons les éléments 
Y_pred=foret.predict(X_test)
print("y_pred",Y_pred)

#verification de la précision de la méthode 
print("Pourcentage de précision Random Forest:",metrics.accuracy_score(Y_test, Y_pred))

#affichage heure de fin 
localtime = time.asctime( time.localtime(time.time()) )
print ("Random Forest heure de fin: ",localtime, "\n")


# affichage de la matrice de confusion
mc(Y_test,Y_pred)