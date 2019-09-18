## Projects
Capstone Project: Starbucks challenge
Focus: Predicting effective offers using app user data

### 1. Installations
This project was coded in Python 3, and was built within the jupyter notebook interface and environment. 
The following Python packages were used:
 - numpy
 - pandas
 - datetime
 - json
 - seaborn
 - calendar
 - matplotlib
 - time
 - sklearn.preprocessing (StandardScaler)
 - sklearn.model_selection (train_test_split, GridSearchCV)
 - sklearn.ensemble
 - sklearn.metrics (classification_report, mean_squared_error)
 - sklearn.linear_model (Ridge)
 - sklearn.tree (DecisionTreeRegressor)
 - sklearn.compose (TransformedTargetRegressor)

### 2. Project Motivation
In this project, we are aiming to predict effective offers to be provided to app users. Depending on the characteristics of an app user, they will be provided with a discount, or BOGO, or informational offer. Ideally, we want to provide app users with the type of offer they are most likely to act on/respond to - ie. an 'effective offer'.

In order to guide the project, we focus on the following business questions:

> 1. For each type of offer, what features (offer characteristics, user characteristics) increases its effectiveness?
> 2. For each type of offer, can we predict the likelihood of an app user responding to the offer?

### 3. File Descriptions
This repo contains one file, called 'Starbucks_Capstone_challenge-MikeBong.ipynb' that contains the code. The data used the project is contained in the folder 'data' within which there are 3 json files: 'portfolio.json', 'profile.json', 'transcript.json'.

### 4. Licensing, Authors, Acknowledgements, etc.
The Starbucks simulated app data was obtained from Udacity and Starbucks. Other references are cited within the notebook.

### 5. Associated pages
Link to Medium post driven by this analysis: TBC
