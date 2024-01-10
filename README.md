# Table of Contents:

1. [Project Goal](#project-goal)
2. [Process](#process)
3. [How to (Install/Run):](#how-to-installrun)



# Project Goal:

The goal of the project is to build an effective model that can predict a users myers briggs personality with relatively good accuraccy based on their last 50 posts on a site called personality cafe. An expanded version of this model also has potential to be useful with social media text data as a whole.


## About Meyers Briggs:

Meyers briggs is a personailty test that groups people into 1 of 16 personailties.
There are four personalities pairs that make up the entire personailtity profile that which acording to meyers briggs, you can only be one or the other.
Meaning that there are 2^4 = 16 possible personalities. 
One can predict someones personality by combining all the predictions for the individual pairs.


More can be learned here:
https://www.myersbriggs.org/my-mbti-personality-type/mbti-basics/ 

## Data Source:

https://www.kaggle.com/datasets/datasnaek/mbti-type

The data file [mbti_1.csv](mbti_1.csv) is a list of users personality types and a corresponding string of their 50 last posts from the personailty cafe forum. Each post is seperated by "|||".


# Process: 

## Transform.py
[Link to view the transform.py program](transform.py)

- The text data of the 50 posts for each user is combined and converted into a sparse term_frequency(tf) matrix of word/term frequencies with column names representing the entire set of words/terms found in at least one users posts.

- The python program (transform.py) transforms the raw data from mbti_1.csv into this tf matrix.

- This matrix along with some columns that represent the users personality type is saved in [tf_matrix.csv](tf_matrix.csv).


## Analysis.py

[Link to view the analysis.py program](analysis.py)

* The second python program (analysis.py) uses the [tf_matrix.csv](tf_matrix.csv) to train a variety of model types to predict the correct personality option for each of the four personality pairs.

* The 4 kinds of models trained and tested to predict each personailty pair are naive bays, logistic regression, decision tree, and random forest

* Since there are 4 model types for each of the 4 personality pairs, there exists 16 models.

* However, this hasn't yet considered  a very sensitive parameter, number of features.

* Too many features can overwelm a model and too little features is insuffient for model profiling and performance.

* Since the entire list of word/term frequencies for all documented words/terms for all users is alot of features for a model, the program uses an optimization technique to reduce the number of features to a more managable and optimal number.

* The best model of a specific type for a certain pair (1 out of 16) is found using optimization, meaning the correct number (or close to best number) of the most influencing features is found.

* The differential_evolution function is what determines the best number of features within an certain integer bounds for a certain model type.

* Multithreading is implemented with the differential_evolution function. Using the same bounds for the number of features, each model type is optimized for the best number of features between those bounds and each of these 4 functions are run together.

* What "together" means in this context is that the program can complete the 4 differential_evolution tasks with partially concurrent execution, allowing usage of a higher percentage of the cpu and speeding up the overall process.


* As state above, there are 16 models for each model type/personality combination
and the best number of features of those 16 model types is found by testing many integer bounds with the differential_evolution function.

* Once the best 16 models have been found, it is time to apply their accuracies to estimate the accuracies of models that can predict the entire personality (all 4 pairs).


* ### Question: How is the accuacy of full personality predictions approximated from individual pairwaise accuracy scores?
    ### Answer: Using the law of independence.

* #### Enforcing independence: 

    * In order to force independence between personality pairs, it is important to note that the same number of users of each personality (39) for each of the 16 personalities is used in the test train process for every model. 

    * There are 624 different users which is 39 users for each of 16 personalities.

    * For each of the four personality pairs, these 624 users are split evenly 50/50 into 312 users of one personality in the pair and 312 users of the opposite personality in the pair

    * When the users are preprocessed (a critical process before each model), the number of test users is 50 and the number of train users is 624-50 = 574

    * With the stratify option, the 50/50 split for any given personality pair of the entire dataset, is maintained for the test users and train users meaning y_train and y_test both have an even split for each personality pair  

- The implications of independence is that the accuracy score of the full myers briggs prediction (one out of 16) can be aproximated by multiplying the accuracy scores of a selected model for each pair. The operation looks like this: (acc_1\*acc_2\*acc_3\*acc_4)

- Besides enforcing independence, the even split of one personality vs the other for a personality pair is critical to build a model with no popularity bias.

- Popularity bias is when the number of occurances of a personalilty in the train data effects how a users personality is predicted by the model.

- No popularity bias means that one personality pair being a certan way does not influence the chance of any other personality pair being a certain way.

- For each of the four model types, the analysis.py program outputs the accuracy results using this consistent model type on each personality pair and combining like above: (acc_1\*acc_2\*acc_3\*acc_4)


- It also outputs the accuracy results of the absolute best model for each personality pair. This means the highest performing model  out of 4 model types is used for each personality type. Again, the accuracy of the entire 4 pair predictions looks like this: (acc_1\*acc_2\*acc_3\*acc_4)

- The outputs are also saved to [results.txt](results.txt)


# How to install/run:

* Requirements:
    * Git
    * python3 and pip (my working python version: 3.10.7)

1. Clone the repository with git.

2. With python3 and pip:
    1. Create a python virtual env in the main project directory. (my working python version: 3.10.7).
    2. Activate the virtual environment
    3. Install the following packages to the virtual env with pip: pandas, nltk, ordered-set, scipy, scikit-learn

3. Run transform.py in the main project directory (and observe results in console).

4. After completion of transform.py, Run analysis.py (and observe results in console).

