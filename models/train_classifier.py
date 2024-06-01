# Import libraries
import sys
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

# Download necessary NLTK data
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
nltk.download('stopwords')

def load_data(database_filepath):
    """
    Load data from SQLite database

    Arguments:
    database_filepath -- string, filepath to the SQLite database

    Returns:
    X -- DataFrame, feature data
    y -- DataFrame, target labels
    category_names -- list, names of target labels
    """
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = os.path.basename(database_filepath).replace(".db", "") + "_table"
    df = pd.read_sql_table(table_name, engine)

    # Extract feature and target variables
    X = df['message']
    columns_to_exclude = ['id', 'message', 'original', 'genre']
    y = df.drop(columns=columns_to_exclude)
    category_names = y.columns.tolist()

    return X, y, category_names

def tokenize(text):
    """
    Tokenize and lemmatize text

    Arguments:
    text -- string, input text

    Returns:
    tokens -- list, processed text tokens
    """
    # Normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

def build_model():
    """
    Build machine learning pipeline and perform grid search

    Returns:
    cv -- GridSearchCV, machine learning model with grid search
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 3, 4],
        'clf__estimator__criterion': ['entropy', 'gini']
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model performance

    Arguments:
    model -- GridSearchCV, trained model
    X_test -- DataFrame, test features
    Y_test -- DataFrame, test labels
    category_names -- list, names of target labels
    """
    y_pred = model.predict(X_test)

    y_pred_df = pd.DataFrame(y_pred, columns=Y_test.columns)

    # Iterate through each target label
    for column in Y_test.columns:
        print('Category: {}\n'.format(column))
        print(classification_report(Y_test[column], y_pred_df[column]))

def save_model(model, model_filepath):
    """
    Save trained model as a pickle file

    Arguments:
    model -- GridSearchCV, trained model
    model_filepath -- string, filepath to save the model
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

def main():
    """
    Main function to orchestrate the machine learning pipeline
    """
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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py data/DisasterResponse.db models/classifier.pkl')

if __name__ == '__main__':
    main()
