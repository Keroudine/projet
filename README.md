# Projet pour la creation d'une application d'orientation accademique

#!/usr/bin/env python
# coding: utf-8

# In[8]:


#shebang


# In[4]:


# Importer les bibliothèques nécessaires
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import zero_one_loss
#import matplotlib.pyplot as plt
#from bs4 import BeautifulSoup
#import seaborn as sns
import pandas as pd
#import numpy as np
#import requests
#import math
import re


# In[5]:


base_Ori_posbac = pd.read_csv('/Users/leprincekerou/Documents/2023-2024/Projet_entreprise/appli_ori/Ori_posbac.csv')


# In[6]:


var_retuenues = ['bac', 'filière' , 'domaine_etu', 'type_lieu_etu', 'pays_bac', 
       'genre', 'Niveau_etu','Col_specialite1','Col_specialite2','Col_matière_prefere2',
      'Col_matière_prefere1', "Col_domaine_activite_pro1"]

# 'satisfact_formation' , 'mention','type_etu','tranche_age',,'type_cours'
df = base_Ori_posbac[var_retuenues]


# In[4]:


import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# Liste des colonnes ordinale à encoder
colonnes_ordinale = ["Niveau_etu"]

df_encoded_ord =df[colonnes_ordinale]
# Liste des colonnes ordinales à encoder
colonnes_ordinale = df_encoded_ord.select_dtypes(include=['object']).columns

# Initialiser l'encodeur ordinal

encoder = OrdinalEncoder()

# Appliquer l'encodage sur toutes les colonnes ordinales
df_encoded_ord[colonnes_ordinale] = encoder.fit_transform(df[colonnes_ordinale])

# Afficher le résultat
df_encoded_ord


# In[5]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder

colonnes_nominale = ['bac', 'type_lieu_etu','genre','domaine_etu',
      'Col_specialite1','Col_specialite2','Col_matière_prefere1',
      'Col_matière_prefere2',"Col_domaine_activite_pro1"]
# Liste des colonnes nominales à encoder
#colonnes_nominale = df.select_dtypes(include=['object']).columns
dn =df[colonnes_nominale]
# Conversion des variables catégorielles en variables indicatrices
df_encoded_nom = pd.get_dummies(dn, columns=colonnes_nominale,drop_first=True)
df_encoded_nom 


# In[6]:


from sklearn.preprocessing import LabelEncoder
# Encodage de la variable cible avec LabelEncoder
label_encoder = LabelEncoder()
df_encoded_nom['filière'] = label_encoder.fit_transform(df['filière'])
#df_encoded("Col_specialite2", axis=1)
df_encoded_nom 


# In[7]:


df_encoded = pd.concat([df_encoded_ord, df_encoded_nom], axis=1)
df_encoded


# In[8]:


X = df_encoded.drop('filière', axis=1)
y = df_encoded['filière']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[9]:


clf = LogisticRegression(penalty='l2',)
clf.fit(X_train, y_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)


# In[10]:


# Évaluation du modèle
err_test = zero_one_loss(y_test, y_pred_test)
err_emp = zero_one_loss(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)
accuracy_train = accuracy_score(y_train, y_pred_train)
print(f"erreur train du modèle : {err_emp}")
print(f"Précision train du modèle : {accuracy_train}")
print("")
print(f"erreur test du modèle : {err_test}")
print(f"Précision test du modèle : {accuracy_test}")


# In[13]:


def var(v):
    return list(base_Ori_posbac[v].unique())

# Fonction modèle
df1 = df
def model(data):
    df = pd.concat([data, df1])
    colonnes_ordinale = ["Niveau_etu"]
    df_encoded_ord = df[colonnes_ordinale].copy()  # Utilisez copy() pour éviter la copie par référence
    
    # Initialiser l'encodeur ordinal
    encoder = OrdinalEncoder()
    
    # Appliquer l'encodage sur toutes les colonnes ordinales
    df_encoded_ord.loc[:, colonnes_ordinale] = encoder.fit_transform(df[colonnes_ordinale])
    
    colonnes_nominale = ['bac', 'type_lieu_etu', 'genre', 'domaine_etu',
          'Col_specialite1', 'Col_specialite2', 'Col_matière_prefere1',
          'Col_matière_prefere2', "Col_domaine_activite_pro1"]
    
    dn = df[colonnes_nominale]
    
    # Conversion des variables catégorielles en variables indicatrices
    df_encoded_nom = pd.get_dummies(dn, columns=colonnes_nominale, drop_first=True)
    
    df_encoded = pd.concat([df_encoded_ord, df_encoded_nom], axis=1)

    

    clf = LogisticRegression(penalty='l2')
    clf.fit(X_train, y_train)
    
    # Exemple avec la première ligne de data, assurez-vous d'ajuster cela en fonction de votre application
    ind = df_encoded.head(1).copy()  # Utilisez copy() pour éviter la copie par référence
    
    y_pred_test = clf.predict(ind)
    y_pred_decoded = label_encoder.inverse_transform(y_pred_test)
    
    return "Vous êtes fait pour les filières de  " + str(y_pred_decoded[0])


# In[16]:


import tkinter as tk
from tkinter import ttk
import pandas as pd

def create_dataframe():
    option1 = dropdown1.get()
    option2 = dropdown2.get()
    option3 = dropdown3.get()
    option4 = dropdown4.get()
    option5 = dropdown5.get()
    option6 = dropdown6.get()
    option7 = dropdown7.get()
    option8 = dropdown8.get()
    option9 = dropdown9.get()
    option10 = dropdown10.get()

    data = {'bac': [option1],'domaine_etu': [option2],'type_lieu_etu': [option3],'genre': [option4],
            'Niveau_etu': [option5],'Col_specialite1': [option6],'Col_specialite2': [option7],'Col_matière_prefere2': [option8],
           'Col_matière_prefere1': [option9],'Col_domaine_activite_pro1': [option10]}
    
    df = pd.DataFrame(data)
    return df


#def display_result():
#    dff = create_dataframe()
#    result = model(dff)
 #   return result_label.config(text=result)


def display_result():
    dff = create_dataframe()
    result = model(dff)
    # Efface le texte précédent
    result_label.config(text="")
    # Affiche le résultat avec une couleur spécifique
    result_label.config(text=result, foreground="blue")  # Vous pouvez choisir la couleur que vous souhaitez




# Crée une fenêtre
window = tk.Tk()
window.title("Application pour s'orienter")
window.geometry("400x650")
c = "SystemButtonFace"
window.configure(bg=c)
# Frame pour organiser les éléments de l'interface
main_frame = ttk.Frame(window)
main_frame.pack(padx=20, pady=10)



# Définition du style pour la combobox
style = ttk.Style()
style.theme_create('custom_theme', parent='alt', settings={
    "TCombobox": {
        "configure": {"foreground": "black", "background": c},
    }
})

#style.theme_use('custom_theme')

style.configure('TLabel', foreground='green', background=c, font=('Arial', 12))

# Crée une liste déroulante pour l'option 1
ttk.Label(window, text="Bac:",style='TLabel').pack()
dropdown1 = ttk.Combobox(window, values=var('bac'), style='Custom.TCombobox')
dropdown1.pack(pady=4, padx=20)

# Crée une liste déroulante pour l'option 2
ttk.Label(window, text="Domaine d'études:").pack()
dropdown2 = ttk.Combobox(window, values=var('domaine_etu'))
dropdown2.pack(pady=4, padx=20)

# Crée une liste déroulante pour l'option 2
ttk.Label(window, text="Type de lieu d'études:").pack()
dropdown3 = ttk.Combobox(window, values=var('type_lieu_etu'))
dropdown3.pack(pady=4, padx=20)

# Crée une liste déroulante pour l'option 2
ttk.Label(window, text="Genre:").pack()
dropdown4 = ttk.Combobox(window, values=var('genre'))
dropdown4.pack(pady=4, padx=20)

# Crée une liste déroulante pour l'option 2
ttk.Label(window, text="Niveau d'études:").pack()
dropdown5 = ttk.Combobox(window, values=var('Niveau_etu'))
dropdown5.pack(pady=4, padx=20)

# Crée une liste déroulante pour l'option 2
ttk.Label(window, text="Spécialité 1:").pack()
dropdown6 = ttk.Combobox(window, values=var('Col_specialite1'))
dropdown6.pack(pady=4, padx=20)


# Crée une liste déroulante pour l'option 2
ttk.Label(window, text="Spécialité 2:").pack()
dropdown7 = ttk.Combobox(window, values=var('Col_specialite2'))
dropdown7.pack(pady=4, padx=20)

# Crée une liste déroulante pour l'option 2
ttk.Label(window, text="Matière préférée 2:").pack()
dropdown8 = ttk.Combobox(window, values=var('Col_matière_prefere2'))
dropdown8.pack(pady=4, padx=20)

# Crée une liste déroulante pour l'option 2
ttk.Label(window, text="Matière préférée 1:").pack()
dropdown9 = ttk.Combobox(window, values=var('Col_matière_prefere1'))
dropdown9.pack(pady=4, padx=20)

# Crée une liste déroulante pour l'option 2
ttk.Label(window, text="Domaine d'activité professionnel:").pack()
dropdown10 = ttk.Combobox(window, values=var("Col_domaine_activite_pro1"))
dropdown10.pack(pady=4, padx=20)


# Créer un style pour le cadre

#style = ttk.Style()
style.configure("ResultFrame.TFrame", background="lightgreen")

# Créer un cadre pour afficher le résultat
result_frame = ttk.Frame(window, style="ResultFrame.TFrame", borderwidth=5, relief="sunken")
result_frame.pack(pady=(10, 20), padx=10, fill="x", expand=True)

# Créer un label pour afficher le résultat dans le cadre
result_label = ttk.Label(result_frame, text="", font=("Arial", 12), wraplength=380)
result_label.pack(pady=10, padx=10)

# Créer un bouton pour afficher les résultats
result_button = tk.Button(window, text="Afficher le résultat", command=display_result,foreground="blue", background=c)
result_button.pack(pady=(5, 15))


window.mainloop()


# In[ ]:




