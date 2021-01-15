import sys


#python train_classifier.py disaster_response.db output_model.wb
#database_filepath = 'disaster_response.db'
#model_filepath = 'output_model.wb'
#print(sys.argv)


# import libraries
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
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidif',TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # specify parameters for grid search
    parameters = {
        #'vect__ngram_range': ((1, 1), (1, 2)),
        #'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__max_features': (None, 5000, 10000),
        #'tfidif__use_idf': (True, False),
        'clf__estimator__n_estimators': [5, 10, 20],
        #'clf__estimator__min_samples_split': [2, 3, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    Y_pred_cat = pd.DataFrame(Y_pred, columns=category_names)

    for i in range(len(category_names)):
        print('Category: {}'.format(category_names[i].upper()), "\n\n",
              classification_report(Y_test.iloc[:,i], Y_pred_cat.iloc[:,i]))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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