import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np
from joblib import numpy_pickle
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from imblearn.metrics import classification_report_imbalanced
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils import resample
events_enhanced = pd.read_csv(r"C:\Users\hmart\OneDrive\Bureau\Cours\Projet\modèles\events_enhanced.csv")
events = pd.read_csv(r"C:\Users\hmart\OneDrive\Bureau\Cours\Projet\dataset\events.csv", sep = ',')
parentid = pd.read_csv(r"C:\Users\hmart\OneDrive\Bureau\Cours\Projet\dataset\category_tree.csv", sep = ',')
properties1= pd.read_csv(r"C:\Users\hmart\OneDrive\Bureau\Cours\Projet\dataset\item_properties_part1.csv", sep = ',')
properties2= pd.read_csv(r"C:\Users\hmart\OneDrive\Bureau\Cours\Projet\dataset\item_properties_part2.csv", sep = ',')
properties = pd.concat([properties1, properties2], ignore_index=True)

pages = ["Présentation", "Visualisation", "Modélisation"]

st.sidebar.title("Navigation")

page = st.sidebar.radio("Aller vers", pages)

if page == pages[0] : 

    st.title("Suivi des utilisateurs de e-commerce sur une période de 4,5 mois")
    st.header("Un projet de pre-processing et de Machine Learning")
    st.write("Vous pouvez retrouver notre projet [ici](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)   ")
    st.subheader("Par Artur Caiano, Yann Ladouceur et Hugo Martinez")

    st.image(r"C:\Users\hmart\OneDrive\Bureau\Cours\Projet\modèles\image.jpg") 

    st.write("A notre disposition nous avons eu les jeux de données suivants : ")
    
    st.subheader("Dataset Events")
    st.dataframe(events.head(10))
    
    st.subheader("Dataset Parentid")
    st.dataframe(parentid.head(10))
    
    st.subheader("Dataset Properties")
    st.dataframe(properties.head(10))

    
    st.header("Première exploration : ")
    st.subheader("Dataset Events")
    


    st.write("Nous avons réalisé plusieurs étapes de pre-processing pour obtenir le DataFrame suivant :")
    st.dataframe(events_enhanced.head(10))

if page == pages[1] :

    fig = plt.figure()

    "insérer ici les visualisations"
    st.pyplot(fig)

if page == pages[2] : 
    "modélisation"
    models = ["RandomForestClassifier"]
    model = st.selectbox("Choissiez votre modèle", models)
    
    if model == models[0]:
        m = joblib.load(r'C:\\Users\\hmart\\OneDrive\\Bureau\\Cours\\Projet\\modèles\\clf_model.joblib')
    