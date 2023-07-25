Goal:

Data Source:
https://www.kaggle.com/datasets/datasnaek/mbti-type

The posts were collected from personailty cafe...


The goal of the project is to predict a users personailty score based on their 50 last post on a site called personailty cafe...

The text data is first conveted into a sparse matrix of 
certain terms occuruances for the user amoung all his posts. 

For each user the term frequency for chosen words is used as the inputs features to each model

There are 4 kinds of models trained to predict each personailty pair: naive bays, logistic regression, decision tree, and random forest

The full personality prediction is the combination of predictions for each of the 4 personality pairs



