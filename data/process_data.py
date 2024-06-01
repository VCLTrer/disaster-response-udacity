import sys
import os
import pandas as pd
from sqlalchemy import create_engine
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(messages_filepath, categories_filepath):
    """
    Load Messages and Categories Data
    
    Arguments:
        messages_filepath: string - Path to the messages CSV file
        categories_filepath: string - Path to the categories CSV file
    
    Output:
        df: DataFrame - Merged DataFrame containing messages and categories data
    """
    try:
        messages = pd.read_csv(messages_filepath)
        categories = pd.read_csv(categories_filepath)
        df = pd.merge(messages, categories, on='id')
        logging.info("Data loaded successfully.")
        logging.debug(df.head())
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        sys.exit(1)

def clean_data(df):
    """
    Clean DataFrame by splitting categories into separate columns and converting values to binary
    
    Arguments:
        df: DataFrame - Combined DataFrame containing messages and categories
    
    Output:
        df: DataFrame - Cleaned DataFrame with separate category columns and binary values
    """
    try:
        # Split categories into separate columns
        categories = df['categories'].str.split(';', expand=True)

        # Extract new column names for categories
        row = categories.iloc[0]
        category_colnames = row.apply(lambda x: x.split('-')[0])
        categories.columns = category_colnames

        # Convert category values to 0 or 1
        for column in categories:
            categories[column] = categories[column].astype(str).str[-1].astype(int)

         # replace 2s with 1s in related column
        categories['related'] = categories['related'].replace(to_replace=2, value=1)    

        # Drop the original categories column from df
        df = df.drop('categories', axis=1)

        # Concatenate the original DataFrame with the new `categories` DataFrame
        df = pd.concat([df, categories], axis=1)

        # Drop duplicates
        df = df.drop_duplicates()

        logging.info("Data cleaned successfully.")
        logging.debug(df.dtypes)
        return df
    except Exception as e:
        logging.error(f"Error cleaning data: {e}")
        sys.exit(1)

def save_data(df, database_filepath):
    """
    Save cleaned data to SQLite database
    
    Arguments:
        df: DataFrame - Cleaned DataFrame
        database_filepath: string - Path to SQLite database file
    """
    try:
        engine = create_engine('sqlite:///' + database_filepath)
        table_name = os.path.basename(database_filepath).replace(".db", "") + "_table"
        df.to_sql(table_name, engine, index=False, if_exists='replace')
        logging.info(f"Data saved to database, table name: {table_name}")
    except Exception as e:
        logging.error(f"Error saving data to database: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        logging.info(f"Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}")
        df = load_data(messages_filepath, categories_filepath)

        logging.info('Cleaning data...')
        df = clean_data(df)

        logging.info(f'Saving data...\n    DATABASE: {database_filepath}')
        save_data(df, database_filepath)
        
        logging.info('Cleaned data saved to database!')
    else:
        logging.error('Please provide the filepaths of the messages and categories '
                      'datasets as the first and second argument respectively, as '
                      'well as the filepath of the database to save the cleaned data '
                      'to as the third argument. \n\nExample: python process_data.py '
                      'disaster_messages.csv disaster_categories.csv '
                      'DisasterResponse.db')

if __name__ == '__main__':
    main()
