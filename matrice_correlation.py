# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 



#lecture des données
data = pd.read_excel('creditcard.xlsx');
X = data[["Time", "V1", "V2", "V3", "V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"]];
Y = data["Class"];

#matrice des corrélations
cmap=plt.cm.Blues #couleur du graph

matrice_correlation= data.corr() 
# fig = plt.figure() 
sns.heatmap(matrice_correlation, vmax = .8, square = True, cmap=cmap)
plt.show()
 