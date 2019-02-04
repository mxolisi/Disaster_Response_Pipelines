import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,how='inner',on=['id'])
    return df

def clean_column_name(column_name):
    """
    Args: column_name: column name to clean
    """
    return re.sub("-\d$","",column_name)



def clean_data(df):
    categories = df['categories'].str.split(';',expand=True)
    categories.columns =df['categories'][0].split(';')
    category_colnames=[]
    for dirty_column in categories.columns:
        category_colnames.append(clean_column_name(dirty_column))
    
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] =  categories[column].map(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    df.drop(['categories'],axis=1,inplace=True)
    df = pd.concat([df,categories],axis=1)
    #Remove duplicates
    df.drop_duplicates(inplace=True)
    return df
    
def save_data(df, database_filename):
    engine = create_engine('sqlite:///{0}'.format(database_filename))
    df.to_sql('InsertTableName', engine, index=False)  


def main():
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