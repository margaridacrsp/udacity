"""
ETL Pipeline
Project: Disaster Response Pipeline (Udacity - Data Science Nanodegree)

Input Arguments:
    -> Path for messages .csv file (e.g. disaster_messages.csv)
    -> Path for the categories .csv file (e.g. disaster_categories.csv)
    -> SQL Database name (e.g. DisasterResponse.db)

"""


#import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories data
    
    Arguments:
        messages_filepath -> Path for the messages .csv file
        categories_filepath -> Path for the categories .csv file

    Outputs:
        df -> DataFrame with the data aggregated    
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories, on='id', how='left')
    
    return df
    

def clean_data(df):
    """
    Clean data 
    
    Arguments:
        df -> Merged messages and categories datasets

    Outputs:
        df -> Merged messages and categories datasets cleaned
    """
    
    #Split column categories in the multiple categories
    categories = pd.Series(df['categories']).str.split(";", expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    #Convert categories values to numbers
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = categories[column].astype(int)
    
    #Drop old categories column from df and concatenate new columns
    df.drop('categories',axis=1,inplace=True)
    df = pd.concat([df,categories],sort=False,axis=1)
    
    #Remove duplicate lines
    df.drop_duplicates(inplace=True)
    
    #Remove the lines where the category 'related' as value 2
    df.drop(df.index[df['related'] > 1],inplace = True)
    
    return df
     

def save_data(df, database_filename):
    """
    Save data to SQL Database
    
    Arguments:
        df -> Merged messages and categories datasets
        database_filename -> SQL Database name
    """
    
    database_filename =  'sqlite:///' + database_filename
    engine = create_engine(database_filename)
    df.to_sql('messages_categories', engine, index = False,if_exists = 'replace')
    


def main():
    """
    Main function that calls the other functions in the following order:
        1. Load messages and categories data
        2. Clean data 
        3. Save data to SQL Database
    """
    
    #Execute the ETL Pipeline if the count of input arguments is 4
    if len(sys.argv) == 4:
        #save the input arguments into variables
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        #1. Load messages and categories data
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        #2. Clean data
        print('Cleaning data...')
        df = clean_data(df)
        
        #3. Save data to SQL Database
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