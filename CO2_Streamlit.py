# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 18:09:42 2021

@author: diogo.da-silva
"""


import streamlit as st

st.title("Démo Streamlit - CarbonPyTracker")



st.markdown("L'objectif de cette démonstration est d'estimer les émissions de CO2 à partir d'un certain nombre de caractéristiques techniques des véhicules.")
st.markdown("Le tableau ci-dessous est un extrait de la base de test.")




import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, LassoCV
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn import metrics
import xgboost as xgb
from xgboost import XGBRegressor
from joblib import dump, load


# Téléchargement et analyse de la base de données nettoyée

from Base_Streamlit import X_test, y_test, feats

# Visualization des features

st.dataframe(X_test.head())


st.markdown("Les deux modèles les plus performants que nous avons testés sur notre base d'entraînement sont le Random Forest Regressor (RFR) et le XGBoost Regressor (XGB).")

st.markdown("Selectionnez le modèle ci-après pour obtenir le score obtenu.")

# Modèles à tester

options=["Random Forest Regressor", "XGBoost Regressor" ]
choix=st.radio("Coefficient de détermination", options=options)

if choix == options[0]:
    RFR_saved=load("RFR.joblib")
    score=RFR_saved.score(X_test, y_test)
    
if choix == options[1]:
    XGB_saved=load("XGB.joblib")
    score=XGB_saved.score(X_test, y_test)
    
st.write("Le score obtenu (R2) est de", score)

st.markdown("Les résultats sont très proches en ce qui concerne le coéfficient de détermination.")
st.markdown("Cependant, le modèle RFR est plus facilement interprétable, car l'importance relative des variables explicatives est plus logique d'un point de vue métier.")



if choix == options[0]:
    RFR_features=pd.DataFrame({"coeffs":RFR_saved.feature_importances_}, index=feats.columns)
    explanatory_variables_RFR=RFR_features.sort_values(by="coeffs", ascending=False).head(10)
    st.dataframe(explanatory_variables_RFR)
    
if choix == options[1]:
    XGB_features=pd.DataFrame({"coeffs":XGB_saved.feature_importances_}, index=feats.columns)
    explanatory_variables_XGB=XGB_features.sort_values(by="coeffs", ascending=False).head(10)
    st.dataframe(explanatory_variables_XGB)


st.markdown("En effet, Le modèle XGB accorde au type de carburant (Diesel ou Essence) une importance qui n'est pas confirmée par la littérature technique.") 
st.markdown("En téorie, La cylindrée, la puissance et la masse des véhicules sont les variables ayant le plus d'impact sur les émissions des véhicules, ce qui nous rassure par rapport aux résultats du modèle RFR.")




