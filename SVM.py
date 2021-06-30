# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import time
from visualisation import matrice_confusion as mc

#permet d'avoir l'heure à laquelle le code commence pour savoir sa durée
localtime = time.asctime( time.localtime(time.time()) )
print ("\nHeure début:",localtime)

#lecture des données
data = pd.read_excel('creditcard.xlsx');
X = data[["Time", "V1", "V2", "V3", "V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"]];
Y = data["Class"];
# print("x",X) # verification qu'on a les bonnes données
# print("y",Y)

# Nous allons maintenant entrainer le code 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#utilisons de la fonction pour le SVM
SVM = svm.SVC(C=10,kernel='rbf',gamma=0.01)

SVM.fit(X_train, y_train)

#Nous prédisons les éléments 
y_pred = SVM.predict(X_test)

#verification de la précision de la méthode
print("Pourcentage de précision SVM:",metrics.accuracy_score(y_test, y_pred))

#affichage heure de fin 
localtime = time.asctime( time.localtime(time.time()) )
print ("SVM heure de fin: ",localtime, "- END \n")


# affichage de la matrice de confusion
# mc(y_test,y_pred)