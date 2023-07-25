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

Process: 

The process involves trainging multiple models on each personailt pair
with the intention being that some models are better suited to different personality pairs. The best model for the entire prediction 1 out of 16 personalities, combines the bets models for eahc pair.

Using this idea, all a pairwise model needs to focus on is the correct pair, 
while the overall model uses the outputs of the pairwise models to make a full prediction.

The point of the optimizers is to focus the model input features to a certain number of terms. Sometimes models with more terms is harder to fit.

...need more about scaling


multithreading is implemented with model optimizers which losely gives the ability to schedule processes in a more efceient way.





Then eahc of 

 uses the best models


How to run/install:

YOu may use the requirements.txtx to install the requireed packages
But it shoud still work with up tp dat version of pyhton and the installed packages