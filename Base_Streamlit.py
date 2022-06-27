# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 12:16:34 2021

@author: diogo.da-silva
"""


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


# Téléchargement et analyse de la base de données

df=pd.read_csv("france.csv")
df.head()


# Supression des variables non-explicatives ou sous-representées 

columns_to_drop=["ID","Country","VFN","Mp","Mh","Man","MMS","Tan","T","Va","Ve","Ct","Cr",\
                 "r","Mt","Ewltp (g/km)","Fm","z (Wh/km)","IT","Ernedc (g/km)",\
                     "Erwltp (g/km)","De","Vf","Status","Electric range (km)",\
                         "year","Electric range (km)"] 


df_main=df.drop(columns_to_drop, axis=1)

df_main.info()

# Rename variables

new_column_names={"Mk":"Marque","Cn":"Modele","m (kg)":"Masse","Enedc (g/km)":"Emissions_CO2","W (mm)":"Empattement",\
             "At1 (mm)":"Largeur_essieu_1","At2 (mm)":"Largeur_essieu_2",\
                 "Ft":"Carburant","ec (cm3)":"Cylindree","ep (KW)":"Puissance"}


df_main=df_main.rename(new_column_names, axis=1)

df_main.info()

# Supression de nan's et duplicates

df_main=df_main.dropna()
df_main.info()

df_main=df_main.drop_duplicates(keep="last")
df_main.info()

# Nettoyage et dichotomisation de la variable "Carburant"

df_main["Carburant"].value_counts()


df_main=df_main.replace(to_replace=["diesel","DIESEL","petrol","PETROL"],\
                        value=["Diesel","Diesel","Petrol","Petrol"])

df_main=df_main.replace(to_replace=["PETROL/ELECTRIC","petrol/electric","Petrol/Electric","Petrol-electric","petrol-electric","Petrol-Electric"],\
                        value=["Petrol","Petrol","Petrol","Petrol","Petrol","Petrol"])   

df_main=df_main.replace(to_replace=["DIESEL/ELECTRIC","diesel/electric","Diesel-electric","Diesel-Electric","Diesel/Electric","diesel-electric"],\
                        value=["Diesel","Diesel","Diesel","Diesel","Diesel","Diesel"])
    
df_main["Carburant"].value_counts()
   
df_main=df_main[(df_main["Carburant"]=="Diesel") | (df_main["Carburant"]=="Petrol")]

df_main["Carburant"].value_counts()   
    

df_main=df_main.join(pd.get_dummies(df_main["Carburant"], prefix="Fuel_Type"))

df_main.info()  


# Display données nettoyées 



# Modélisation

feats=df_main.drop(["Emissions_CO2","Carburant","Marque","Modele"], axis=1)
target=df_main["Emissions_CO2"]

X_train, X_test, y_train, y_test= train_test_split(feats, target, test_size=0.2)

# Random Forest Regressor

RFR = ensemble.RandomForestRegressor()
RFR.fit(X_train, y_train)
dump(RFR,"RFR.joblib")

# XGBoost Regressor

XGB=XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)

XGB.fit(X_train,y_train)
dump(XGB,"XGB.joblib")













     
    
    
