# This project consists of three parts:

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

# How to run/test the project :
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
