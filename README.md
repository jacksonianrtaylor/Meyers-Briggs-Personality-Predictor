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


About Meyers Briggs:

Meyers briggs is a personailty test that groups people into 1 of 16 personailties
There are four personalities pairs that make up the entire personailtity profile that which acording to m eyers briggs you can be one or the other
Meaning that there are 2^4 = 16 possible personalities
One can predict personality by combining all the predictions for the individual pairs
More can be learned here:
https://www.myersbriggs.org/my-mbti-personality-type/mbti-basics/




Process: 

The process involves training multiple models on each personality pair
with the intention being that some models are better suited to predict different personality pairs. The best model for the entire prediction 1 out of 16 personalities, combines the best models for each pair.

Using this idea, all a pairwise model needs to focus on is the correct selection for a single pair, while an overall model that indicated full personailty uses the outputs of the pairwise models to make a full prediction.

The point of the optimizers is to focus the model input features to a certain number of terms. Sometimes models with more terms is harder to fit.

The count of each


multithreading is implemented with model optimizers which losely gives the ability to schedule processes in a more efceient way.







How to run/install:

install a certin version of python with venv creation...

https://stackoverflow.com/questions/1534210/use-different-python-version-with-virtualenv
https://docs.python.org/3/library/venv.html



You may use the requirements.txt to install the require=d package versions
But it shoud still work with the most up to date version of python and the installed packages




