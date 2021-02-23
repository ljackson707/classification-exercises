# Imports
from env import host_name, user_name, password
from pydataset import data
import pandas as pd 
import os

# Helper Function (To gain connection to SQL database)
def get_connection(db, user=user_name, host=host_name, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

# Make a function named get_titanic_data that returns the titanic data from the codeup 
# data science database as a pandas data frame. Obtain your data from the Codeup Data 
# Science Database.

def get_titanic_data(cached=False):
    '''
    This function reads the titanic data from the Codeup db into a df,
    write it to a csv file, and returns the df.
    '''
    # Create SQL query.
    sql_query = 'SELECT * FROM passengers'
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('titanic_db'))
    
    return df

# Make a function named get_iris_data that returns the data from the iris_db on the codeup 
# data science database as a pandas data frame. The returned data frame should include the 
# actual name of the species in addition to the species_ids. Obtain your data from the Codeup 
# Data Science Database.

def get_iris_data(cached=False):
    '''
    This function reads the titanic data from the Codeup db into a df,
    write it to a csv file, and returns the df.
    '''
    # Create SQL query.
    sql_query = 'SELECT * FROM species'
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('iris_db'))
    
    return df

# Once you've got your get_titanic_data and get_iris_data functions written, now it's time to 
# add caching to them. To do this, edit the beginning of the function to check for a local 
# filename like titanic.csv or iris.csv. If they exist, use the .csv file. If the file doesn't 
# exist, then produce the SQL and pandas necessary to create a dataframe, then write the 
# dataframe to a .csv file with the appropriate name. 

# Converting titanic_db to local csv
    df.to_csv('titanic_df.csv')
    os.path.isfile('titanic_df.csv')

# Converting iris_db to local csv
    df.to_csv('iris_df.csv')
    os.path.isfile('iris_df.csv')
