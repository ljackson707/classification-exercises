# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import aquire 
from aquire import get_iris_data

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def clean_iris(df):
    '''
    clean_iris will take in one arg. df a pandas dataframe, anticipated to be the iris dataset
    and will remove species_id and measurement_id columns,
    rename species_name to species
    encode species into two new columns
    
    return a single pandas dataframe with the above operations preformed
    '''
    dropcols = ['species_id', 'measurement_id']
    df.drop(columns = dropcols, inplace = True)
    df.rename(columns = {'species_name':'species'}, inplace = True)
    dummies = pd.get_dummies(df[['species']], drop_first = True)
    return pd.concat([df, dummies], axis = 1)

def prep_iris(df):
    '''
    prep_iris will take one arg. df, a pandas dataframe, anticipated to be the iris dataset 
    and will remove species_id and measurement_id columns,
    rename species_name to species
    encode species into two new columns
    
    preform a train, validate, and test split
    
    return three pandas dataframes, train, validate, test
    '''
    df = clean_iris(df)
    # test 
    train_validate, test = train_test_split(df, test_size = 0.2, random_state= 1349, stratify = df.species)
    # train and validate
    train, validate = train_test_split(train_validate, train_size=.70, random_state=1349, stratify=train_validate.species)
    return train, validate, test
