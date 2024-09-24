# Disaster-Response-Classifier
This project is about natural language processing of Disaster-Response messages and to categorize them with the help of a Random Forest Classifier. It helps people or organization in an event of a disasterto to classify emergency messages into 36 different categories. The project contains a pipeline for text processing, training a classifier for text classification, to store a model and a webapp to calssify Disaster Response text into categories.

### How to run the Python scripts
 Run in a command prompt 
- process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
- train_classifier.py data/DisasterResponse.db models/classifier.pkl

### How to run the web app 
1. Run the following commands in the project's root directory to set up your database and model in a commad prompt.

    - To run ETL pipeline that cleans data and stores in database
        `.../process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `.../train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Go to `app` directory and Run your web app: `python run.py`


### Files in the repository
Folder app
- run.py
Folder model
- train_classifier.py
Folder data
- process_data.py
- messages.csv
- categories.csv

Jupyter Notebooks as py and html
- ETL Pipeline Preparation.py
- ETL Pipeline Preparation.html
- ML Pipeline Preparation.py
- ML Pipeline Preparation.html


### Libraries which are used
- numpy 
- pandas
- matplotlib.pyplot 
- csv
- sys
- nltk
- nltk.download('punkt')
- nltk.tokenize word_tokenize
- nltk.tokenize sent_tokenize
- numpy as np
- pandas as pd
- matplotlib.pyplot as plt
- sklearn
- csv
- os
- sqlalchemy  create_engine
- sklearn.ensemble RandomForestClassifier
- sklearn.multioutput MultiOutputClassifier
- sklearn.pipeline Pipeline
- sklearn.metrics classification_report
- sklearn.model_selection train_test_split
- sklearn.feature_extraction.text TfidfVectorizer
- sklearn.model_selection train_test_split, GridSearchCV
- pickle
- json
- plotly
- Flask
- flask render_template, request, jsonify
- plotly.graph_objs Bar
- sklearn.externals joblib

