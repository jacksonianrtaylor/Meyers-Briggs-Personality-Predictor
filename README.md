# Table of Contents:

1. [Project Goal](#project-goal)
2. [Process](#process)
3. [How to (Install/Run):](#how-to-installrun)



# Project Goal:

The goal of the project is to theorize an effective model that can predict a users Myers Briggs personality with relatively good accuracy based on their last 50 posts on a site called personality cafe. An expanded version of this model also has potential to be useful with users social media text data regardless of the source.


## About Myers Briggs:

Myers Briggs is a personality test that groups people into 1 of 16 personalities.
There are four personalities pairs that make up the entire personality profile that which according to Myers Briggs, you can only be one or the other.
Therefore, there are 2^4 = 16 possible personalities. 
One can predict someones personality by combining all the predictions for the individual pairs.

More can be learned here:
https://www.myersbriggs.org/my-mbti-personality-type/mbti-basics/ 

## Data Source:

https://www.kaggle.com/datasets/datasnaek/mbti-type

The data file [mbti_1.csv](mbti_1.csv) is a list of users personality types and a corresponding string of their 50 last posts from the personality cafe forum. Each post is separated by "|||".


# Process: 

## transform.py
[Link to view the transform.py program](transform.py)

- The text data of the 50 posts for each user is combined and converted into a sparse term frequency-inverse document frequency (tf-idf) matrix with column names for the entire set of words/terms found in at least one users posts.

- The python program (transform.py) transforms the raw data from mbti_1.csv into this tf-idf matrix.

- This matrix along with some columns that represent the users personality type is saved in [tf_idf_matrix.csv](tf_idf_matrix.csv).


## analysis.py

[Link to view the analysis.py program](analysis.py)

* The second python program (analysis.py) uses the [tf_idf_matrix.csv](tf_idf_matrix.csv) to train a variety of model types to predict the correct personality option for each of the four personality pairs.

* The 3 kinds of models trained and tested to predict each personality pair are Logistic Regression, Random Forest, and Naive Bayes.

* Since there are 3 model types that can be tuned for each of the 4 personality pairs, there exists 12 models.

* The features that are tuned are the words/terms features. Each of these terms represent a column of tf-idf scores (one for each user) from which you can either use or ignore.

* Too many features can overwhelm a model and too little features is insufficient for model profiling and performance. 

* Since the entire list of documented words/terms for all users is a lot of features for a model, the program uses an optimization technique to reduce the number of features to a more manageable and optimal number.

* The best model of a specific type for a certain pair (1 out of 12) is found using optimization, meaning the correct number (or close to best number) of the most influencing features is found.

* The differential_evolution optimizer is what determines the best performing number of features within an certain integer bounds (40,60) for a certain (model type, personality pair) combination.

* The only notable expense in the program occurs in each of three calls to differential_evolution for the three model types inside the personality pair loop.

* The number of workers for a differential evolution call are set too the cpu_count of the machine to max out cpu resources in this task.

* Once the best 12 models have been found corresponding to each (model type, personality pair), it is time to apply their accuracies to estimate the accuracies of models that can predict the entire personality (all 4 pairs).


* ### Question: How is the accuracy of full personality predictions approximated from individual pairwise accuracy scores?
    ### Answer: Using the law of independence.

* #### Enforcing independence: 

    * In order to force independence between personality pairs, it is important that the same number of users of each personality (39) for each of the 16 personalities (624 total) is used in the train-test process for every model. 

    * This means that for each of the four personality pairs, the 624 users are split evenly (50/50) into 312 users of one personality in the pair and 312 users of the opposite personality in the pair

    * Using StratifiedKFold in the cross validation function, the split of both personality pairs stays even for both train and test users. 


- The implications of independence between personality pairs is that the accuracy score of the full Myers Briggs prediction (1 out of 16) can be approximated by multiplying the accuracy scores of a selected model for each pair. The operation looks like this: (acc_1\*acc_2\*acc_3\*acc_4)

- Besides enforcing independence, the even split of one personality vs the other for a personality pair is critical to build a model with no popularity bias.

- Popularity bias is when the percentage of occurrences of a personality in the train data effects how a users personality is predicted by the model.

- The absence of popularity bias means that one personality pair being a certain way does not influence the chance of any other personality pair being a certain way.

- For each of the three model types, the analysis.py program outputs the best model accuracy results using 4 of the 12 models that are the corresponding model type where each model contains the accuracy of a predicting a distinct personality pair. The accuracy of the entire 4 pair predictions look like this: (acc_1\*acc_2\*acc_3\*acc_4)

- It also outputs the accuracy results of the absolute best model for each personality pair. This means the highest performing model out of 3 model types is used for each personality pair. Again, the accuracy of the entire 4 pair predictions looks like this: (acc_1\*acc_2\*acc_3\*acc_4)

- The outputs are also saved to [results.txt](results.txt)


# How to install/run:

* Requirements:
    * Git
    * python3 and pip (my working python version: 3.10.7)

1. Clone the repository with git.

2. With python3 and pip:
    1. Create a python virtual env in the main project directory. (my working python version: 3.10.7).
    2. Activate the virtual environment
    3. Install the following packages to the virtual env with pip: pandas, numpy, nltk, scipy, scikit-learn

3. Run transform.py in the main project directory (and observe results in console).

4. After completion of transform.py, Run analysis.py (and observe results in console or results.txt).


