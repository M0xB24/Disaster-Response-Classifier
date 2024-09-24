import sys
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import csv
import os
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
import pickle


def load_data(database_filepath):
    """
    Loads a sqlite db which should be created with process_data.py and stores the data in a pandas df
    
    """
    # load data from database
    engine = create_engine(f'sqlite:///{os.path.abspath(database_filepath)}')
    df = pd.read_sql_table('CleanedData',con=engine)  
    X = df['message'].astype(str)
    Y = df.drop(columns=['message', 'original', 'id', 'genre']).astype(str)
    return  X, Y


def tokenize(text):
    """
    Part of natural laguage processing, tokenize and lemmatize text input and returns it
    
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Create a pipeline to tokenize and to create a random forest classifier
    
    """
    pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=tokenize ,max_df=0.95, min_df=5)),  # Convert text to numerical features
    ('clf', MultiOutputClassifier(RandomForestClassifier(max_depth=None, n_estimators=100)))  # Multi-output classification
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test):
    """
    Helper method to evaluate a trained model
    
    """    

    Y_pred = model.predict(X_test)

    for i, col in enumerate(Y_test.columns):
        print(f"Category: {col}")
        score = classification_report(Y_test.iloc[:, i], Y_pred[:, i])
        print(score)
    return score


def save_model(model, model_filepath):
    """
    Save a created model into a pkl file for later use of the model
    
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

    print("Model saved as best_model.pkl")


def main():
    """
    Load data, build a pipeline, train the model, evaluate a model and save it.
    
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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