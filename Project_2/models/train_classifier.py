"""
ML Pipeline
Project: Disaster Response Pipeline (Udacity - Data Science Nanodegree)

Input Arguments:
    -> SQL Database name (e.g. DisasterResponse.db)
    -> Path to pickle file where the final model will be saved(e.g. classifier.pkl)

"""

# import libraries
import sys

import pandas as pd
import numpy as np
import pickle
import re
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt','stopwords','wordnet'])
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    """
    Load data from database
    
    Arguments:
        database_filepath -> Path of SQL Database

    Outputs:
        X -> dataframe containing the features
        Y -> dataframe containing the labels
        category_names -> List of categories names
    """
    
    #inicialize database
    database_filepath =  'sqlite:///' + database_filepath
    engine = create_engine(database_filepath)
    
    #load table and create model variables
    df = pd.read_sql_table('messages_categories',engine)
    X = df['message']
    Y = df.drop(columns = ['id','message','original','genre'])
    category_names = list(Y.columns)
    
    return X,Y,category_names


def tokenize(text):
    """
    Tokenize function
    
    Arguments:
        text -> Input text

    Outputs:
        words_lemmed -> Text tokenized
    """
    
    #Normalize text: remove pontuation + lower text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #Tokenize text
    words = text.split()
    
    #Remove stop words
    #words = [w for w in words if w not in stopwords.words("english")]
    
    #Lemmatize
    words_lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]

    return words_lemmed


def build_model():
    """
    Build pipeline model 
    
    Outputs:
        cv -> ML Pipeline that process text messages and apply a classifier
    """
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidif',TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # specify parameters for grid search
    parameters = {
        'clf__estimator__n_estimators': [10, 20]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate pipeline model 
    
    Arguments:
        model -> ML Pipeline
        X_test -> Test features
        Y_test -> Test labels
        category_names -> List of categories names
    """
    
    #Predict Model
    Y_pred = model.predict(X_test)
    Y_pred_cat = pd.DataFrame(Y_pred, columns=category_names)
    
    #Create classification report with: f1 score, precision and recall for each output category of the dataset
    for i in range(len(category_names)):
        print('Category: {}'.format(category_names[i].upper()), "\n\n",
              classification_report(Y_test.iloc[:,i], Y_pred_cat.iloc[:,i]))


def save_model(model, model_filepath):
    """
    Save pipeline model 
    
    Arguments:
        model -> ML Pipeline
        model_filepath -> Path to save the model in a .pkl file
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    Main function that calls the other functions in the following order:
        1. Load data from SQL database
        2. Train ML model on training set
        3. Evaluate Model
        4. Save model in .pkl file
    """
    
    #Execute the ML Pipeline if the count of input arguments is 3
    if len(sys.argv) == 3:
        #save the input arguments into variables
        database_filepath, model_filepath = sys.argv[1:]
        
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        #1. Load data from SQL database
        X, Y, category_names = load_data(database_filepath)
        
        #2. Train ML model on training set
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print('Building model...')
        model = build_model() 
        print('Training model...')
        model.fit(X_train, Y_train)
        
        #3. Evaluate Model
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        
        #4. Save model in .pkl file
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()