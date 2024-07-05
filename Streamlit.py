# Commande a lancer depuis power shelle prompt de Anaconda
# cd 'C:\Users\fta\Desktop\98-DATASCIENTEST\SPRINT3\117-Streamlit\Exo'
# streamlit run Streamlit.py
# Directement  
# streamlit run c:/Users/fta/Desktop/98-DATASCIENTEST/SPRINT3/117-Streamlit/Exo/Streamlit.py'

# importer la librairie Streamlit et les librairies d'exploration de données et de DataVizualization nécessaires.
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  confusion_matrix

import os 
script_dir = os.path.dirname(__file__)  # Répertoire du script
file_path = os.path.join(script_dir, "train.csv")

# Créer un dataframe appelé df permettant de lire le fichier train.csv
df = pd.read_csv(file_path)


# Copier le code suivant pour ajouter un titre au Streamlit et créer 3 pages appelées "Exploration", "DataVizualization" et "Modélisation".
st.title("Projet de classification binaire Titanic")
st.sidebar.title("Sommaire")
pages=["Exploration", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)


if page == pages[0] : 
# Ecrire "Introduction" en haut de la première page 
  st.write("### Introduction")
# Afficher les 10 premières lignes du dataframe
  st.dataframe(df.head(10))
# Afficher des informations sur le dataframe
  st.write(df.shape)
  st.dataframe(df.describe())

# La méthode df.info() écrit dans la console, nous devons rediriger la sortie pour l'afficher correctement
  buffer = io.StringIO()
  df.info(buf=buffer)
  s = buffer.getvalue()
  st.text(s)


# Créer une checkbox pour choisir d'afficher ou non le nombre de valeurs manquantes
  if st.checkbox("Afficher les NA") :
    st.dataframe(df.isna().sum())


if page == pages[1] : 
# Ecrire "DataVizualization" en haut de la deuxième page
  st.write("### DataVizualization")

# Afficher dans un plot la distribution de la variable cible
  fig = plt.figure()
  sns.countplot(x = 'Survived', hue='Survived', data = df)
  st.pyplot(fig)

#  Afficher des plots permettant de décrire les passagers du Titanic
  fig = plt.figure()
  sns.countplot(x = 'Sex', hue='Sex', data = df)
  plt.title("Répartition du genre des passagers")
  st.pyplot(fig)

  fig = plt.figure()
  sns.countplot(x = 'Pclass', data = df)
  plt.title("Répartition des classes des passagers")
  st.pyplot(fig)

  fig = sns.displot(x = 'Age', data = df)
  plt.title("Distribution de l'âge des passagers")
  st.pyplot(fig)

# Afficher un countplot de la variable cible en fonction du genre
  fig = plt.figure()
  sns.countplot(x = 'Survived', hue='Sex', data = df)
  st.pyplot(fig)

# Afficher un plot de la variable cible en fonction des classes
  fig = sns.catplot(x='Pclass', y='Survived', data=df, kind='point')
  st.pyplot(fig)

# Afficher un plot de la variable cible en fonction des âges  
  fig = sns.lmplot(x='Age', y='Survived', hue="Pclass", data=df)
  st.pyplot(fig)

# Sélectionner uniquement les colonnes numériques
# Calculer la corrélation
# Afficher la matrice de corrélation
  numeric_df = df.select_dtypes(include=['number']) 
  correlation_matrix = numeric_df.corr() 
  st.write("### Matrice de corrélation")
  st.dataframe(correlation_matrix)
  
# Ecrire "Modélisation" en haut de la troisième page
if page == pages[2] : 
  st.write("### Modélisation") 

# supprimer les variables non-pertinentes (PassengerID, Name, Ticket, Cabin)
  df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# créer une variable y contenant la variable target. 
  y = df['Survived']
# créer un dataframe X_cat contenant les variables explicatives catégorielles
  X_cat = df[['Pclass', 'Sex',  'Embarked']]
# créer un dataframe X_num contenant les variables explicatives numériques
  X_num = df[['Age', 'Fare', 'SibSp', 'Parch']]

# remplacer les valeurs manquantes des variables catégorielles par le mode et remplacer les valeurs manquantes des variables numériques par la médiane
  for col in X_cat.columns:
    X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
  for col in X_num.columns:
    X_num[col] = X_num[col].fillna(X_num[col].median())

# encoder les variables catégorielles
  X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)
# concatener les variables explicatives encodées et sans valeurs manquantes pour obtenir un dataframe X clean
  X = pd.concat([X_cat_scaled, X_num], axis = 1)

# séparer les données en un ensemble d'entrainement et un ensemble test
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# standardiser les valeurs numériques
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
  X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])

# créer une fonction appelée prediction qui prend en argument le nom d'un classifieur et renvoie le classifieur entrainé
  def prediction(classifier):
    if classifier == 'Random Forest':
        clf = RandomForestClassifier()
    elif classifier == 'SVC':
        clf = SVC()
    elif classifier == 'Logistic Regression':
        clf = LogisticRegression()
    clf.fit(X_train, y_train)
    return clf
  

# il est intéressant de regarder l'accuracy des prédictions
  def scores(clf, choice):
    if choice == 'Accuracy':
        return clf.score(X_test, y_test)
    elif choice == 'Confusion matrix':
        return confusion_matrix(y_test, clf.predict(X_test))
    

# utiliser la méthode st.selectbox() pour choisir entre le classifieur RandomForest, le classifieur SVM et le classifieur LogisticRegression
  choix = ['Random Forest', 'SVC', 'Logistic Regression']
  option = st.selectbox('Choix du modèle', choix)
  st.write('Le modèle choisi est :', option)

  clf = prediction(option)
  display = st.radio('Que souhaitez-vous montrer ?', ('Accuracy', 'Confusion matrix'))
  if display == 'Accuracy':
      st.write(scores(clf, display))
  elif display == 'Confusion matrix':
      st.dataframe(scores(clf, display))

 