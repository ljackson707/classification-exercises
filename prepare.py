# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import aquire 
from aquire import get_iris_data

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


df = get_iris_data()
df.info()

def clean_iris():
    dropcols = ['species_id']
    df.drop(columns = dropcols, inplace = True)
    df.rename(columns = {'species_name':'species'}, inplace = True)
    dummies = pd.get_dummies(df[['species']], drop_first = True)
    return pd.concat([df, dummies], axis = 1)

def impute_mode():
    '''
    impute mode for species 
    '''
    imputer = SimpleImputer(strategy='most_frequent')
    train[['species']] = imputer.fit_transform(train[['species']])
    validate[['species']] = imputer.transform(validate[['species']])
    test[['species']] = imputer.transform(test[['species']])
    return train, validate, test

def prep_iris_data():
    train, validate, test = train_test_split(df, test_size=.50, random_state=123, stratify=df.species)
    train, validate = train_test_split(train_validate,
                                       test_size=.50, 
                                       random_state=123, 
                                       stratify=train_validate.species)
    train, validate, test = impute_mode()
    return train, validate, test
