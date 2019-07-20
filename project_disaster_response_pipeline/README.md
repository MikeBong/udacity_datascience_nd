# Disaster Response Pipeline Project
-----
### Files:
There are 4 sub folders within this project folder as follows:
- data -> contains ETL pipeline files to pre-process, clean and save data
- models -> contains file to train and save model
- app -> contains file to render flask web app
- screenshots -> contains screen grabs of the landing page and message classification
The contents of each folder is elaborated further below.
---
#### 'data' folder
This folder contains the raw input data in csv ('disaster_categories.csv', 'disaster_messages.csv'). 
It also contains the 'process_data.py' file that pre-processes, cleans and saves data in an SQLite database. The default name for the database is 'DisasterResponse.db'.

#### 'models' folder
This folder contains the 'train_classifier.py' file that loads the processed data from the SQLite database, trains a selected natual language processing classification model (currently using AdaBoostClassifier), and exports the trained model into a pickle file. The default name for the trained model 'classifier_model_ada.pkl'.

#### 'app' folder
This folder contains the 'run.py' file which uses the contents of the 'template' sub-folder (contains html files) to render the flask web app. Running the 'run.py' script will load data and the trained model for use in the web app. The resultant web app will then display charts using the loaded data and enable the app to take in 'messages'and classify them based on the trained model, and present classification outputs. 

#### 'screenshots' folder
This folder contains screen grabs of the main landing page which charts out the insights from the loaded data. It also contains the classifiction output for the message: "Help! There was an earthquake and there are many missing, injured and dead people, we need some medicine for treatment!".

![alt text](https://github.com/MikeBong/udacity_datascience_nd/blob/master/project_disaster_response_pipeline/screenshots/01_into_main_landing_page.png)

![alt text](https://github.com/MikeBong/udacity_datascience_nd/blob/master/project_disaster_response_pipeline/screenshots/02_1_message_classification_pt01.png)
![alt text](https://github.com/MikeBong/udacity_datascience_nd/blob/master/project_disaster_response_pipeline/screenshots/02_2_message_classification_pt02.png)
-----
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier_model_ada.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

-----
### References:
- https://www.datacamp.com/community/tutorials/adaboost-classifier-python
