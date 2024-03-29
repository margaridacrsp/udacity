import sys
import pandas as pd
from sqlalchemy import create_engine

#python process_data.py disaster_messages.csv disaster_categories.csv disaster_response.db
#messages_filepath='disaster_messages.csv'
#categories_filepath = 'disaster_categories.csv'
#database_filepah = 'disaster_response.db'
#print(sys.argv)

def load_data(messages_filepath, categories_filepath):
    #load messages and categories data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #merge the two dataset in one
    df = messages.merge(categories, on='id', how='left')
    
    return df
    

def clean_data(df):
    #split column categories in the multiple categories
    categories = pd.Series(df['categories']).str.split(";", expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    #convert categories values just to numbers
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = categories[column].astype(int)
    categories.head()
    
    #drop old categories column from df and concatenate new columns
    df.drop('categories',axis=1,inplace=True)
    df = pd.concat([df,categories],sort=False,axis=1)
    
    #remove duplicate lines
    df.drop_duplicates(inplace=True)
    
    return df
     

def save_data(df, database_filename):
    database_filename =  'sqlite:///' + database_filename
    engine = create_engine(database_filename)
    df.to_sql('messages_categories', engine, index=False)
    


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