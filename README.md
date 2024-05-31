This project consists of three parts:

1. ETL Pipeline
process_data.py writes a data cleaning pipeline that:

Loads the messages and categories datasets
Merges the two datasets 
Cleans the data
Stores it in a SQLite database

2. ML Pipeline
train_classifier.py writes a machine learning pipeline that:

Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file

3. Flask Web App
Displays visualizations about the training data using Plotly
Runs the model on new messages that you enter yourself