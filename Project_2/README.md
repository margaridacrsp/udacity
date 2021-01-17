### Disaster Response Pipeline Project

This project consists of classifying messages, in time of disaster, into certain categories.
For this, a model was built based on the data provided by FigureEight, where for various messages it is indicated to which categories, of the 36 available, each message belongs.

The model was compiled into a web application, where it is possible to insert a new message and see which categories the mesage falls into. In the web application, is also displayed some visualizations of the data used to train the model


This project was developed under the Udacity Data Science Nanodegree program.


### Overview

This project is divided in three steps:
    1. Load the data from FigureEight and clean it in an ETL Pipeline;
    2. Write a ML Pipeline and train it and optimize it with the data provided;
    3. Provide a message and some data visualization in an web application.


### Files

Data folder:
    - disaster_categories.csv : CSV file containing the 36 possible categories
    - disaster_messages.csv : CSV file containing the disaster messages
    - DisasterResponse.db : Database where the final dataset from the ETL Pipeline is saved
    - process_data.py : ETL Pipeline, where the data is load, cleaned and saved into the database
    - ETL Pipeline Preparation.ipynb : training jupyter notebook to development the ETL Pipeline

Models folder:
    - train_classifier.py : ML Pipeline, where the is built and trained
    - classifier.pkl : ML Pipeline model output
    - ML Pipiline Preparation.ipynb : training jupyter notebook to development the ML Pipeline
    
App folder:
    - run.py : web application launcher 
    - templates: where the .html files of the two pages of the web applications are saved


### Run

In the console navigate to the top-level project directory and run commands in the following sequence:
    - python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    - python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
    - python app/run.py
    
Then open the web application trow the http://0.0.0.0:3001/ url.


### Web Application

![Alt text](screenshots/master_page.jpg?raw=true)

![Alt text](screenshots/go_page.jpg?raw=true)