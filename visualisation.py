# -*- coding: utf-8 -*-
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns 

def matrice_confusion(y_test,y_pred):
    cmap=plt.cm.Blues #mettre la couleur du graph en bleau
    LABELS = ['Normal', 'Fraude'] 
    conf_matrix = confusion_matrix(y_test, y_pred) #on définit les élément à prendre pour faire notre matrice de confusion 
    plt.figure(figsize =(12, 12)) 
    sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, fmt ="d",cmap=cmap); #on crée le graph
    plt.title("matrice de confusion") 
    plt.ylabel('Classe réelle') 
    plt.xlabel('Classe prédite') 
    plt.show() 
    
