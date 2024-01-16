import pandas as pd
import time
import copy
import random
from scipy.sparse import csr_matrix
from scipy.optimize import differential_evolution
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# from scipy.optimize import aneal


from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


# from threading import Thread
import multiprocessing

# SEED_INT = 10


def select_model(classifier):
    if (classifier=="naive_bays_model"):
        return MultinomialNB()
    elif(classifier=="log_reg_model"):
        return LogisticRegression(max_iter = 1000)
    elif(classifier=="rand_forest_model"):
        return RandomForestClassifier()
    elif(classifier == "dec_tree_model"):
        return DecisionTreeClassifier()




    # LOOK: implement cross-validation

    # 624 users (574 train, 50 test):
    # This can be repeated up to 12.48 -> 12 times with non-overlapping test sets

    # idea queue:

    # Question: Where does the feature selector go?

    # step 0: Train a feature selector
    # then, use the feature selector to reduce the features for all X.

    # this is already done...

    # step 1: shuffle the users (features and personalities as pairs)
    # step 2: split into groups like above
    # step 3: run the model on each group and return the average.


    # LOOK: If applying the trained model after training process on a single data point,...
    # then you need to use feature selection on this test data that same way it is used in the training
    # The problem is that chi-squred feature selection requires more than one test user to implement
    # If the exact feature were known then those feature could simply be chosen, but they differ between folds.
    # However, if large batch was tested instead of a single user, then the same feature selection process can be applied...
    # for the best given number of features.


class feature_selection_classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, nof_features, classifier_id):

        self.nof_features = nof_features
        self.classifier_id = classifier_id

        #LOOK: Should model selection be defined here???
        self.selector = SelectKBest(chi2, k=self.nof_features)
        self.model = select_model(self.classifier_id)

    def fit(self, X, y):
        # Check that X and y have correct shape
        # This has been removed to keep csr matrices
        # X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = np.unique(y)

        # Your custom fitting logic goes here....
        self.selector.fit(X, y)
        X_transformed = self.selector.transform(X)
        self.model.fit(X_transformed, y)

        # Mark the estimator as fitted
        self._is_fitted = True

        return self

    def predict(self, X):
        # Check if the estimator is fitted
        check_is_fitted(self, '_is_fitted')

        # Input validation
        # This has been removed to keep csr matrices
        # X = check_array(X)

        # Your custom prediction logic goes here
        X_transformed = self.selector.transform(X)
        predictions = self.model.predict(X_transformed)

        return predictions



def test_features(i, classifier_id, X, y,accuracy_dict,nof_features_dict,val):
    """Train and test a model after preprocessing:"""

    #steps: 
    #1. create an estimator object
    #2. create a StratifiedKFold object
    #3. use cross_val_score

    nof_features = round(i[0])

    model = feature_selection_classifier(nof_features, classifier_id)
    skfold = StratifiedKFold(n_splits=3, shuffle=True)

    #LOOK: try increasing number of jobs here to increase cpu usage
    #this may avoid the need for multiprocessing.
    #this can be altered to fit a persona machine
    scores = cross_val_score(model, X, y, cv=skfold, scoring='accuracy', n_jobs=1)

    average_score = sum(scores)/len(scores)
    accuracy_dict[val] = average_score
    nof_features_dict[val] = nof_features
    
    return -average_score




def main():

    start_time = time.time()
    f = open('tf_matrix.csv', 'r', encoding="utf-8")
    data = pd.read_csv(f, header=0) 

    # Personality pairs used for outputs:
    pairs  =  ["_I_E_","_N_S_", "_T_F_", "_J_P_"] 

    # Features are all word/tokens that occur in at least one of the users posts (based on "word_bank" in the transfrom.py programm).
    features = list(data.keys())

    features = features[:-4]

    # X is the tf matrix from tf_matrix.csv (not the personality data)
    X = data[features]
    X = X.values

    # convert to X csr matrix becuase the numpy array is sparse
    X = csr_matrix(X)

 

    # The eventual shape of best_in_class is [4][2][4].
    # The first dimension is the personality pair being tested.
    # The seconds dimension is two lists.

    # The first a list of accuracies scores for each model type.
    # The second list is the corresponding optimal number of features for each model type. 

    # "best_in_class" is used to theoreticly compute the accuracy of any single classifer type predicting the full personality (4 types correctly).
    # "best_in_class" is also used to theoreticlly compute the accuracy when the absolute best optimized classfier type...
    # is used to predict each personailty pair (4 types correctly).

    best_in_class = []

    print("Training and testing with (39 X 16) = 624 users for each personality pair\n")


    for item in pairs:  
        mini_start = time.time()
        # Target values of the current personality pair:
        y = list(data[item])
          
        # The optimiztion function, differential_evolution, finds the best value between the bounds of the number of features that maximizes model performance.
        # Multithreading is used here for some improvements in runtime.

        #LOOK: try multiples runs without seeding the optimizer  
        #does
        manager = multiprocessing.Manager()
        accuracy_dict = manager.dict()
        nof_features_dict = manager.dict()


        #LOOK: I can use multiple sets bounds for the same test function here

        # t1 = multiprocessing.Process(target=differential_evolution, kwargs={"func" : test_features, "bounds" : [(20, 25)], "args" : ("dec_tree_model", copy.deepcopy(X), copy.deepcopy(y), accuracy_dict,nof_features_dict, 0), "seed" : 5})

        # t2 = multiprocessing.Process(target=differential_evolution, kwargs={"func" : test_features, "bounds" : [(20, 25)], "args" : ("log_reg_model", copy.deepcopy(X), copy.deepcopy(y), accuracy_dict,nof_features_dict, 1), "seed" : 5})

        # t3 = multiprocessing.Process(target=differential_evolution, kwargs={"func" : test_features, "bounds" : [(20, 25)], "args" : ("rand_forest_model", copy.deepcopy(X), copy.deepcopy(y), accuracy_dict,nof_features_dict, 2), "seed" : 5})

        # t4 = multiprocessing.Process(target=differential_evolution, kwargs={"func" : test_features, "bounds" : [(20, 25)], "args" : ("naive_bays_model", copy.deepcopy(X), copy.deepcopy(y), accuracy_dict,nof_features_dict, 3), "seed" : 5})

        # t1 = multiprocessing.Process(target=differential_evolution, kwargs={"func" : test_features, "bounds" : [(20, 28)], "args" : ("log_reg_model", copy.deepcopy(X), copy.deepcopy(y), accuracy_dict,nof_features_dict, 1), "seed" : 5})

        # t2 = multiprocessing.Process(target=differential_evolution, kwargs={"func" : test_features, "bounds" : [(20, 22)], "args" : ("rand_forest_model", copy.deepcopy(X), copy.deepcopy(y), accuracy_dict,nof_features_dict, 2), "maxiter" : 3, "seed" : 5})

        # t3 = multiprocessing.Process(target=differential_evolution, kwargs={"func" : test_features, "bounds" : [(23, 25)], "args" : ("rand_forest_model", copy.deepcopy(X), copy.deepcopy(y), accuracy_dict,nof_features_dict, 3), "maxiter" : 3, "seed" : 5})

        # t4 = multiprocessing.Process(target=differential_evolution, kwargs={"func" : test_features, "bounds" : [(26, 28)], "args" : ("rand_forest_model", copy.deepcopy(X), copy.deepcopy(y), accuracy_dict,nof_features_dict, 4), "maxiter" : 3, "seed" : 5})

        
        # # t1.start()
        # t2.start()
        # t3.start()
        # t4.start()  

        # # t1.join()
        # t2.join()
        # t3.join()
        # t4.join()   

        #without multi-processing
        differential_evolution(func = test_features, bounds =  [(20, 28)], args =  ("rand_forest_model", copy.deepcopy(X), copy.deepcopy(y), accuracy_dict, nof_features_dict, 0),
                                maxiter = 9, seed = 5, workers = 3)

        accuracies = list(accuracy_dict.values())
        nof_features = list(nof_features_dict.values())

        print(accuracies)
        print(nof_features)
        print((time.time() - mini_start)/60, "minutes")

    print("Full compute time:",float((time.time() - start_time)/60), "Minutes")
            
            
#         # Print the accuracy and the number of features for the best of each classifier for the given pair.
#         print(item, "Classification:")
#         print("Best Predictor Function Scores:")
#         print("Decision Tree:","Accuracy:", accuracies[0],"Features:",nof_features[0])
#         print("Logistic Regession:","Accuracy:",accuracies[1],"Features:",nof_features[1])
#         print("Random Forest:", "Accuracy:",accuracies[2],"Features:",nof_features[2])
#         print("Naive Bays:", "Accuracy:",accuracies[3],"Features:",nof_features[3],"\n")
        

#         best_in_class.append([accuracies, nof_features])


#     # Classifier types used for outputs:
#     classifiers = ["Decision Tree", "Logistic Regression", "Random Forest", "Naive Bays"]

#     print("Complete Myers Briggs Prediction for individual Classifiers:\n")

#     f = open("results.txt","w")

#     f.write("Complete Myers Briggs Prediction for individual Classifiers:")
#     f.write("\n\n")

#     # Find the best models that uses the same classifer type for each personality pair and output results.
#     for i in range (0,4):
#         pair_wise_features = ""
#         model_accuracy =1
#         j = 0
#         for pair in best_in_class:
#             model_accuracy*= pair[0][i]
#             pair_wise_features+= "Pair: "+pairs[j]+" Features: "+str(pair[1][i])+" "
#             j+=1
            
#         print("Classifier:", classifiers[i])
#         print("Accuracy:",str(model_accuracy))
#         print(pair_wise_features,"\n")
#         f.write("Classifier: "+classifiers[i])
#         f.write("\n")
#         f.write("Accuracy: "+str(model_accuracy))
#         f.write("\n")
#         f.write(pair_wise_features)
#         f.write("\n\n")



#     # Find the absolute best model:
#     best_classifer_and_nof_features_per_personailty_pair = []
#     best_model_accuracy  = 1

#     i = 0
#     for pair in best_in_class:
#         classifier_acc_nof_features = list(zip(classifiers, pair[0], pair[1]))
#         classifier_max_acc_nof_features = max(classifier_acc_nof_features, key = (lambda x: x[1]))
#         best_model_accuracy*= classifier_max_acc_nof_features[1]
#         best_classifer_and_nof_features_per_personailty_pair.append("Pair: "+str(pairs[i])+\
#                                                                     " Classifier: " +str(classifier_max_acc_nof_features[0])+\
#                                                                     " Number of features: "+str(classifier_max_acc_nof_features[2]))
#         i+=1


#     # Print and output the best model accuracy and the top models for each personality pair.
#     f.write("Myers Briggs Prediction from the best of each Classifier:")
#     f.write("\n")
#     f.write("Best model accuracy: " + str(best_model_accuracy))
#     f.write("\n")

#     print("Myers Briggs Prediction from the best of each Classifier:")
#     print("Best model accuracy:", best_model_accuracy)

#     for item in best_classifer_and_nof_features_per_personailty_pair:
#         f.write(item)
#         f.write("\n")
#         print(item)
#     f.close()


#     # Computation time:
#     print("Done")
#     print("Full compute time:",float((time.time() - start_time)/60), "Minutes")



if __name__ == "__main__":
    main()


