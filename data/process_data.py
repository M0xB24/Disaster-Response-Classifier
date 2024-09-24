import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import csv
import os
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):

    """
    Loads the data from two csv files and merge them together accordinng to shared id categories.
    
    """
    messages = pd.read_csv(messages_filepath, sep=',',encoding='utf-8')  
    categories = pd.read_csv(categories_filepath, sep=',',encoding='utf-8')
    df = pd.merge(messages, categories, on='id', how='inner')
    # Split the categories column on ';' to create multiple columns
    categories_new  =  pd.DataFrame()
    categories_new = df['categories'].str.split(';', expand=True)
    # Extract category names by taking the part before '-0' or '-1'
    row = categories_new.iloc[0]  # Get the first row (can be any row)
    category_colnames = row.apply(lambda x: x.split('-')[0])
    # Rename the columns
    categories_new.columns = category_colnames
    # Convert category values to 0 or 1 by taking the last character of each string
    for column in categories_new:
        # set each value to be the last character of the string
        categories_new[column] = categories_new[column].apply(lambda x: int(x.split('-')[1]))
    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories_new], axis=1)
    return df


def clean_data(df):
    """
    Drops duplicates in the pandas dataframe
    
    """
    df_no_duplicates = df.drop_duplicates()
    df_cleaned = df_no_duplicates[df_no_duplicates['related'] != 2]
    return df_cleaned



def save_data(df, filepath):
    """
    Saves the loaded and processed pandas df to a sqlite database
    
    """
    engine = create_engine(f'sqlite:///{os.path.abspath(filepath)}')
    df.to_sql('CleanedData', engine, index=False, if_exists='replace')  


def main():
    """
    Main method to import, clean and save data
    
    """

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()