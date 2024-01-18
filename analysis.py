import time
import copy
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from scipy.optimize import differential_evolution

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted



def select_model(classifier):
    '''
    Simple classifier model type selection by input string.
    Any featureSelectionClassifier object works similarly, no matter which type of model is chosen. 
    '''
    if(classifier=="log_reg_model"):
        return LogisticRegression(max_iter = 1000)
    elif(classifier=="rand_forest_model"):
        return RandomForestClassifier()
    elif (classifier=="naive_bayes_model"):
        return MultinomialNB()


class featureSelectionClassifier(BaseEstimator, ClassifierMixin):
    '''
    Derived from BaseEstimator and ClassifierMixin to work as an estimator.
    Model is two parts: feature selection and a classifier. 
    '''
    def __init__(self, nof_features, classifier_id):

        self.nof_features = nof_features
        self.classifier_id = classifier_id

        self.selector = SelectKBest(chi2, k=self.nof_features)
        self.model = select_model(self.classifier_id)

    def fit(self, X, y):
        # Check that X and y have correct shape...
        # This has been removed to allow csr matrices
        # X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = np.unique(y)

        # Model fitting logic:
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
        # This has been removed to allow csr matrices
        # X = check_array(X)

        # Prediction logic:
        X_transformed = self.selector.transform(X)
        predictions = self.model.predict(X_transformed)

        return predictions


def test_model(i, classifier_id, X, y):
    """Train and test with cross validation"""

    # Steps: 
    #1. Create an estimator object
    #2. Create a StratifiedKFold object
    #3. Use cross_val_score to train and test

    nof_features = round(i[0])

    model = featureSelectionClassifier(nof_features, classifier_id)
    skfold = StratifiedKFold(n_splits=12, shuffle=True)

    scores = cross_val_score(model, X, y, cv=skfold, scoring='accuracy', n_jobs=1)

    average_score = sum(scores)/len(scores)
    
    return -average_score



def main():

    start_time = time.time()
    f = open('tf_idf_matrix.csv', 'r', encoding="utf-8")
    data = pd.read_csv(f, header=0) 

    # Personality pairs used for outputs:
    pairs  =  ["_I_E_","_N_S_", "_T_F_", "_J_P_"] 

    # Features are all word/tokens that occur in at least one of the users posts (based on "terms" in the transform.py program).
    features = list(data.keys())

    features = features[:-4]

    # X is the tf-idf from tf_idf_matrix.csv (not the personality data)
    X = data[features]
    X = X.values

    # convert X to csr matrix because the numpy array is sparse and setting this improves performance
    X = csr_matrix(X)


    # The eventual shape of best_in_class is [4][2][3].
    # The first dimension is the personality pair being tested.
    # The seconds dimension is two lists.

    # The first is a list of accuracies scores for each model type. 
    # The second list is the corresponding optimal number of features for each model type. 

    # "best_in_class" is used to theoretically compute the accuracy of any single classifier type predicting the full personality (4 types correctly).
    # "best_in_class" is also used to theoretically compute the accuracy when the absolute best optimized classifier type...
    # is used to predict each personality pair (4 types correctly).

    best_in_class = []

    print("Training and testing with (39 X 16) = 624 users\n")


    for item in pairs:  
        # Target values of the current personality pair:
        y = list(data[item])
          
        # The list of the best accuracies for the personality pair (item for each model type)
        acc_list = []
        # The corresponding best input to the number of word/term features 
        nof_features_list = []

        # The optimization function, differential_evolution, finds the best value between the bounds of the number of features that maximizes model performance.
        # The number of workers becomes equal to the cpu cores available on the system...
        # So cpu resources are maxed out to find the optimal number of features for the given model type.
        res1 = differential_evolution(func = test_model, bounds =  [(40, 60)], args =  ("log_reg_model", copy.deepcopy(X), copy.deepcopy(y)),
                                maxiter = 8, seed = 5, updating = "deferred", workers = -1, x0 = [60])
        acc_list.append(-res1.fun)
        nof_features_list.append(round(res1.x[0])) 
        print("log_reg_model optimized for",item)   
        res2 = differential_evolution(func = test_model, bounds =  [(40, 60)], args =  ("rand_forest_model", copy.deepcopy(X), copy.deepcopy(y)),
                                maxiter = 8, seed = 5, updating = "deferred", workers = -1, x0 = [60])
        acc_list.append(-res2.fun)
        nof_features_list.append(round(res2.x[0]))  
        print("rand_forest_model optimized for",item)       
        res3 = differential_evolution(func = test_model, bounds =  [(40, 60)], args =  ("naive_bayes_model", copy.deepcopy(X), copy.deepcopy(y)),
                                maxiter = 8, seed = 5, updating = "deferred", workers = -1, x0 = [60])
        acc_list.append(-res3.fun)
        nof_features_list.append(round(res3.x[0]))  
        print("naive_bayes_model optimized for",item)  


        # Print the accuracy and the number of features for the best of each classifier for the given pair.
        print()
        print(item, "Classification:")
        print("Best Predictor Function Scores:")
        print("Logistic Regression:","Accuracy:",acc_list[0],"Features:",nof_features_list[0])
        print("Random Forest:", "Accuracy:",acc_list[1],"Features:",nof_features_list[1])
        print("Naive Bayes:", "Accuracy:",acc_list[2],"Features:",nof_features_list[2],"\n")
        

        best_in_class.append([acc_list, nof_features_list])


    # Classifier types used for outputs:
    classifiers = ["Logistic Regression", "Random Forest", "Naive Bayes"]

    print("Complete Myers Briggs Prediction for individual Classifiers:\n")

    f = open("results.txt","w")

    f.write("Complete Myers Briggs Prediction for individual Classifiers:")
    f.write("\n\n")

    # Find the best models that uses the same classifier type for each personality pair and output results.
    for i in range (0,3):
        pair_wise_features = ""
        model_accuracy =1
        j = 0
        for pair in best_in_class:
            model_accuracy*= pair[0][i]
            pair_wise_features+= "Pair: "+pairs[j]+" Features: "+str(pair[1][i])+" "
            j+=1
            
        print("Classifier:", classifiers[i])
        print("Accuracy:",str(model_accuracy))
        print(pair_wise_features,"\n")
        f.write("Classifier: "+classifiers[i])
        f.write("\n")
        f.write("Accuracy: "+str(model_accuracy))
        f.write("\n")
        f.write(pair_wise_features)
        f.write("\n\n")



    # Find the absolute best model:
    best_classifier_and_nof_features_per_personality_pair = []
    best_model_accuracy  = 1

    i = 0
    for pair in best_in_class:
        classifier_acc_nof_features = list(zip(classifiers, pair[0], pair[1]))
        classifier_max_acc_nof_features = max(classifier_acc_nof_features, key = (lambda x: x[1]))
        best_model_accuracy*= classifier_max_acc_nof_features[1]
        best_classifier_and_nof_features_per_personality_pair.append("Pair: "+str(pairs[i])+\
                                                                    " Classifier: " +str(classifier_max_acc_nof_features[0])+\
                                                                    " Number of features: "+str(classifier_max_acc_nof_features[2]))
        i+=1


    # Print and output the best model accuracy and the top models for each personality pair.
    f.write("Myers Briggs Prediction from the best of each Classifier:")
    f.write("\n")
    f.write("Best model accuracy: " + str(best_model_accuracy))
    f.write("\n")

    print("Myers Briggs Prediction from the best of each Classifier:")
    print("Best model accuracy:", best_model_accuracy)

    for item in best_classifier_and_nof_features_per_personality_pair:
        f.write(item)
        f.write("\n")
        print(item)
    f.close()


    # Computation time:
    print("Done")
    print("Full compute time:",float((time.time() - start_time)/60), "Minutes")



if __name__ == "__main__":
    main()


