# Goal:

The goal of the project is to predict a users personality score based on their last 50 posts on a site called personality cafe.


# Data Source:

https://www.kaggle.com/datasets/datasnaek/mbti-type

The posts were originally collected from personailty cafe and formated into this collection.


# About Meyers Briggs:

Meyers briggs is a personailty test that groups people into 1 of 16 personailties.
There are four personalities pairs that make up the entire personailtity profile that which acording to meyers briggs, you can only be one or the other.
Meaning that there are 2^4 = 16 possible personalities. 
One can predict someones personality by combining all the predictions for the individual pairs.


More can be learned here:
https://www.myersbriggs.org/my-mbti-personality-type/mbti-basics/ 


# Process: 

## Transform.py

- The text data of the 50 posts for each user is combined and converted into a sparse tf matrix of word/term frequencies with column names representing the entire set of words/terms found in at least one users posts.

- The first py file in the project (transform.py) transforms the raw data from mbti_1.csv into this tf matrix 

- This matrix along with some columns that represent the users personality type is saved in tf_matrix.csv


### Analysis.py

- The second py file (analysis.py) trains a variety of models to predict the correct personality option for each pair.

* Note: This means that for each model type, there are  4 different sub models a that are specialized to predict a certain pair.

* The 4 kinds of models trained and tested to predict each personailty pair are naive bays, logistic regression, decision tree, and random forest

* Since there are four personality pairs for each of 4 modelt types, there exists 16 models.

* However, this hasn't yet considered  a very sensitive parameter, number of features...

* Too many features can overwelm a model and too little features is insuffient for model profliing.

* Since the entire list of word/term frequencies for all documented words/terms for all users is alot of features for a model, the analysis.py uses an optimization technique to reduce the number of features to a more managable and optimal number

* The best model for a specific type is found for each personality pair using optimization, meaning the correct number of the most influencing features (number of words/terms) is used

* Multithreading is implemented with model optimizers which losely gives the ability to the cpu to schedule processes in a more efficient way, decreasing the runtime 

* Since there are 4 personality pairs and 4 model categories, this means there are 16 unqiue models if there is to be a single best model with an optimal number of features for each pair-model type combination

* The full personality prediction (one out of 16) is the combination of predictions for each of the 4 personality pairs 

* Using this idea, all a pairwise model needs to focus on is the correct selection for a single pair, while an overall model that indicates the full personailty uses the outputs of the pairwise models to make a full prediction.

* The best model for the entire prediction of one out of 16 personalities, combines the best models for each pair.

---


- It is important to note that the same number of users of each personality (39) for each of the 16 personalities is
used in the test train process for every model. 

- There are 624 different users which is 39 users for each of 16 personalities

- For each of the four personality pairs, these 624 users are split evenly 50/50 into 312 users of one personality in the pair and 312 users of the opposite personality in the pair

- When the users are preprocessed (a critical process before each model), the number of test users is 50 and the number of train users is 624-50 = 574

- With the stratify option, the 50/50 split for any given personality pair of the entire dataset, is maintained for the test users and train users
meaning y_train and y_test both have an even split for each personality pair  

- The even split of one personality vs the other for a personality pair is critical to build a model wih no popularity bias

- popularity bias is when the number of occurance of a personalilty in the train data effects how a users personality is predicted by the model

- no popularity bias means that one personality pair being a certan way does not influence the chance of any other personality pair being a certain way

- The effect of no popularity bias, is that the training and testing for any given pair can be independent from the training and testing any other personality pair, and the prediction of the complete personality (4 different pairs) can be achived by applying each model.

- Whats more, the accuracy score of the full myers brggs prediction (one out of 16) can be aproximated by muliplying the accuracy scores of each distinct pairwise model 

- The analysis.py program, applies and prints the accuracy results of a single model type, trained and tested for each personality type and feature optimzed for that personailty pair,

- It also prints the accuracy results of the absolute best model for each personality pair (the best feature usage for the best model type)







misc:

- This also means that the full data set can be used to train/test every personality pair

- If the models were used to simply predict the entire personality (1 out of 16) then data is not reused for each pair and there is less training potential (it would require a larger dataset)


- The process involves training multiple models on each personality pair
with the intention being that some models are better suited to predict different personality pairs. The best model for the entire prediction 1 out of 16 personalities, combines the best models for each pair.

- Using this idea, all a pairwise model needs to focus on is the correct selection for a single pair, while an overall model that indicated full personailty uses the outputs of the pairwise models to make a full prediction.

- (LOOK) this is not exactly right...
on average the chosen train data is split 50/50 but there is variation.
Solution: stratify the outputs

...



How to install/run:

Quick Way (with docker):

requirements:
-Git
-Dockerdesktop
-Sufficent memory

clone the repository
navigate to the main project directory
build docker image using the provided Dockerfile:
run the created docker image:
observe outputs from the console of the running image
computation can take some time (estimation)



Explicit Way (no docker):

requirements:
-Git
-Sufficent memory
-python3 and pip (my working python version: 3.10.7)

clone the repository

with python and pip:
create a python virtual env in the main project directory:
install the following packages to the virtual env:
scipy, scikit-learn, pandas, nltk, ordered_set
activate the virtual environment
use python command to run the transform.py (and observe results in console)
after completion of transform.py, run analysis.py (and observe results in console)

computation can take some time (estimation)



