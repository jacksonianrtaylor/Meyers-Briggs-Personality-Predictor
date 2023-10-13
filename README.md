Data: 

Data Source:
https://www.kaggle.com/datasets/datasnaek/mbti-type

The posts were collected from personailty cafe...


About Meyers Briggs:

Meyers briggs is a personailty test that groups people into 1 of 16 personailties
There are four personalities pairs that make up the entire personailtity profile that which acording to m eyers briggs you can be one or the other
Meaning that there are 2^4 = 16 possible personalities
One can predict personality by combining all the predictions for the individual pairs
More can be learned here:
https://www.myersbriggs.org/my-mbti-personality-type/mbti-basics/ 



Goal:

The goal of the project is to predict a users personailty score based on their last 50 posts on a site called personailty cafe.

the input data to the model for a given user is a list of word/term counts in a consistent order that at least one user in the entire data set posted

The text data of the 50 posts for each user is combined and converted into a sparse matrix of term frequencies 

Since the entire list of word/term frequencies is alot of features for a model the analysis.py uses an optimization technique to reduce the number of features to a optimal number

There are 4 kinds of models trained to predict each personailty pair: naive bays, logistic regression, decision tree, and random forest

The best model for each of these classes is foudn for each personality pair (this means)

since there are 4 personality pairs, this means there are 16 relevant models if there is to be a single best model with an optimal number of featurs for each pair model combination





The full personality prediction is the combination of predictions for each of the 4 personality pairs 




This is done with individual models to predict every pair as well as the combination of the best performing models for each pair

when indiviudal models are used 





Process: 

The process involves training multiple models on each personality pair
with the intention being that some models are better suited to predict different personality pairs. The best model for the entire prediction 1 out of 16 personalities, combines the best models for each pair.

Using this idea, all a pairwise model needs to focus on is the correct selection for a single pair, while an overall model that indicated full personailty uses the outputs of the pairwise models to make a full prediction.

The point of the optimizers is to focus the model input features to a certain number of terms. Sometimes models with more terms is harder to fit.

multithreading is implemented with model optimizers which losely gives the ability tot the cpu to schedule processes in a more efficient way.


It is important to note that the same number of users of each personmality (39) for each of the 16 personalities is
used in the test train process for each model. 

This measn when any model predicts a pair their is no popularity bais buid into the model. 

(LOOK) this is not exactly right...
on average the chosen train data is split 50/50 but there is variation.
Solution: stratify the outputs
...


#it is important to note that the personailty pair prediction should not have an impact on 
#another personality pair since there is the same number of people for each of the 16 personalilties

#This eliminates bais for common vs uncommon personalities
#it means that a full prediction is not made but is theoreticaly accurate based on the predictions of each pair
#using the law of independence and the best scored model for each personality pair...




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



