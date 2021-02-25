#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import aquire

from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder

from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For titanic data

def prep_titanic_data(df):
    df = df\
        .pipe(handle_missing_values)\
        .pipe(remove_columns)\
        .pipe(encode_embarked)
    return df

def handle_missing_values(df):
    return df.assign(
        embark_town=df.embark_town.fillna('Other'),
        embarked=df.embarked.fillna('O'),
    )

def remove_columns(df):
    return df.drop(columns=['deck'])

def encode_embarked(df):
    encoder = LabelEncoder()
    encoder.fit(df.embarked)
    return df.assign(embarked_encode = encoder.transform(df.embarked))

def train_validate_test_split(df, seed=123):
    train_and_validate, test = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df.survived
    )
    train, validate = train_test_split(
        train_and_validate,
        test_size=0.3,
        random_state=seed,
        stratify=train_and_validate.survived,
    )
    return train, validate, test

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For iris data 

def prep_iris_data(df):
    df = df\
        .pipe(remove_columns_i)
    return df

def remove_columns_i(df):
    return df.drop(columns=['species_id', 'measurement_id'])

def train_validate_test_split_i(df, seed=123):
    train_and_validate, test = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df.species_name
    )
    train, validate = train_test_split(
        train_and_validate,
        test_size=0.3,
        random_state=seed,
        stratify=train_and_validate.species_name,
    )
    return train, validate, test

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Split function

def split(df, stratify_by=None):
    """
    Crude train, validate, test split
    To stratify, send in a column name
    """
    
    if stratify_by == None:
        train, test = train_test_split(df, test_size=.3, random_state=123)
        train, validate = train_test_split(df, test_size=.3, random_state=123)
    else:
        train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[stratify_by])
        train, validate = train_test_split(df, test_size=.3, random_state=123, stratify=train[stratify_by])
    
    return train, validate, test

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Idk if I can us this anymore

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