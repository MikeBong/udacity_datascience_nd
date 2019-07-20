import sys
import pandas as pd

from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet'])

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

import pickle


def load_data(database_filepath):
    """
    Function: To load data from SQLite database.
    
    Input:
        database_filepath -> path to SQLite database in string ''
    
    Output:
        X -> feature data frame
        Y -> label data frame
        category_names -> column names used in data viz app
    """
    engine = create_engine('sqlite:///'+database_filepath)
#     engine = create_engine('sqlite:///'+'../data/'+database_filepath)
    df = pd.read_sql_table(database_filepath, engine)
    
    X = df.loc[:,'message']
    Y = df.iloc[:, 4:]
    category_names = list(Y.columns)
    
    return X, Y, category_names


def tokenize(text):
    """
    Function: To tokenize and process text data.
    
    Input:
        text -> text data to be cleaned and/or tokenized
    
    Output:
        clean_tokens -> cleaned, tokenized text for ML modelling
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
    Function: Returns model containing machine learning pipeline.
    
    Input:
        NA    
    Output:
        generate a machine learning pipeline, with parameters to be fed into the model
    """    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize,ngram_range=(1,2),max_df=1.0)),
        ('tfidf', TfidfTransformer(use_idf=True)),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])   

# Note: Ran to idetify the best param combo, commented out as it's a lengthy process. 
#       Have passed in output of cv.best_params_ into pipeline above.
#     parameters = {
#         'vect__ngram_range': ((1, 1), (1, 2)),
#         'vect__max_df': (0.5, 1.0),
#         'tfidf__use_idf': (True, False)
#     }
#     cv = GridSearchCV(pipeline, param_grid=parameters)
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function: Returns model classification report.
    
    Input:
        model -> trained model
        X_test -> fields/features in test dataset fed into model 
        Y_test -> categories/output in test dataset evaluated by model
        category_names -> name of output categories to be classified
    Output:
        trained model's classification reports 
    """     
    y_pred = model.predict(X_test)
    for a,b in enumerate(category_names):
        print("For column {}:".format(b))
        print(classification_report(Y_test[b], y_pred[:,a]))


def save_model(model, model_filepath):
    """
    Function: To save/export model as a pickle file.
    
    Input:
        model -> final trained model
        model_filepath -> filepath location to export model as pickle file
    
    Output:
        saved pickle file of model in designated location
    """
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


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