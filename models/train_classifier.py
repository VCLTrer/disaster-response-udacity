# import libraries
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
from sklearn.metrics import classification_report, accuracy_score, fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV
import pickle

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
nltk.download('stopwords')

def load_data(database_filepath):
    
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    table_name = os.path.basename(database_filepath).replace(".db","") + "_table"
    print(table_name)
    df = pd.read_sql_table(table_name,engine)

    #Remove child alone as it has all zeros only
    #df = df.drop(['child_alone'],axis=1)

    X = df['message']
    columns_to_exclude = ['id','message','original','genre']
    y = df.drop(columns=columns_to_exclude)
    #y.head()
    category_names = y.columns.tolist()

    return X, y, category_names


def tokenize(text):
    """ Tokenize text to used in pipeline """
    
    #normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]"," ", text.lower())
    
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    #tokenize text
    tokens = word_tokenize(text)
    
    #lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word)for word in tokens if word not in stop_words]
    return tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {'clf__estimator__n_estimators': [50, 100],
              'clf__estimator__min_samples_split': [2, 3, 4],
              'clf__estimator__criterion': ['entropy', 'gini']
              }
    #scorer = make_scorer(fbeta_score, beta=2)
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    # predict on test data
    y_pred = model.predict(X_test)

    #report
    y_pred_df = pd.DataFrame(y_pred, columns = Y_test.columns)

    # Iterate through the columns of the dataset
    for column in Y_test.columns:
        print('column_name: {}\n'.format(column))
        print(classification_report(Y_test[column], y_pred_df[column]))
   

def save_model(model, model_filepath):

    #new_file_name = 'classifier.pkl'
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
     
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
        #print(Y_train[].dtypes)
        
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