import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Function: To load data from messages.csv, categories.csv, merge both datasets and return on key ('id').
    
    Input:
        messages_filepath -> path to messages dataset
        categories_filepath -> path to categories dataset
    
    Output:
        df -> merged dataset of messages and categories
    """    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = pd.merge(messages, categories, left_on='id', right_on='id')
    
    return df


def clean_data(df):
    """
    Function: To pre-process and clean data:
        -> Generate category column names
        -> Fill in new category columns with correct inputs (1,0)
        -> Modify incorrect values (2 into 1)
        -> Add modified category data to df
        -> Remove duplicates
    
    Input:
        df -> df to be enhanced with category data and cleaned
    
    Output:
        df -> df that has been augmented with category data and cleaned
    """      
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # Additional step to replace all 2's into 1's
    categories = categories.replace(2,1)
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    return df   


def save_data(df, database_filename):
    """
    Function: Save df to a SQLite database
    
    Input:
        df -> df to be saved
        database_filename -> database name and file path
    
    Output:
        saved database
    """  
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql(database_filename, engine, index=False, if_exists='replace')


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