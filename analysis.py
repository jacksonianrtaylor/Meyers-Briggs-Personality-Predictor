
import pandas as pd
import time

import random
from threading import Thread

from scipy.sparse import csr_matrix
from scipy.optimize import fminbound

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Global random seed for consistent train test splits and randomstates for certain models:
SEED_INT = 10


class return_thread(Thread):
    """Thread class that returns a value with join:"""
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

# Classification models:
# They all return accuracy scores (the number of correct predictions divided by the number of predictions).

def naive_bays_model(X_train,X_test,y_train,y_test):
    naive_bays = MultinomialNB()
    naive_bays.fit(X_train,y_train)
    y_pred=naive_bays.predict(X_test)   
    return accuracy_score(y_test, y_pred)

def log_reg_model(X_train,X_test,y_train,y_test):
    log_reg = LogisticRegression(max_iter = 1000)
    log_reg.fit(X_train,y_train)
    y_pred=log_reg.predict(X_test)   
    return accuracy_score(y_test, y_pred)

def rand_forest_model(X_train,X_test,y_train,y_test):
    rand_for = RandomForestClassifier(random_state = SEED_INT)
    rand_for.fit(X_train,y_train)
    y_pred=rand_for.predict(X_test)   
    return accuracy_score(y_test, y_pred)

def dec_tree_model(X_train,X_test,y_train,y_test):
    """The decision tree model implements cost complexity pruning."""
    dec_tree = DecisionTreeClassifier(random_state = SEED_INT)
    path = dec_tree.cost_complexity_pruning_path(X_train, y_train)
    alphas = path['ccp_alphas']   
    max_score = 0
    for alpha in alphas:
        score = fit_tree(X_train,X_test,y_train,y_test, alpha)
        if score>max_score:
            max_score = score                  
    return max_score  

def fit_tree(X_train,X_test,y_train,y_test, alpha):
    """Get the accuracy score of a specific tree with the given "alpha":"""
    if(alpha < 0):
        alpha = 0
    dec_tree_a = DecisionTreeClassifier(random_state = SEED_INT, ccp_alpha = alpha)
    dec_tree_a.fit(X_train,y_train)
    y_pred = dec_tree_a.predict(X_test)   
    return accuracy_score(y_test, y_pred)
    
   


class tests():   
    """
    This is what is used to test the accuracy of models for varied number of feature inputs...
    that are selected with chi-squared feature selection.
    The input features are simply word counts.
    """
    def __init__(self):
        # This is used for consistent train test splits for a given runtime for all models.
        # However, it has been replaced with a constant seed to produce similair consistency over runtimes.
        # SEED_INT is used instead.
        self.seed = random.randint(0, 42) 

    def preprocess(self, i,X, y):
        """Transform the data to be ready for a model:"""
        #  Stratify makes sure the same proportions of target values in the full dataset is preseved in the train and test data 
        #  Both the y_train and y_test are split 50/50 between any given pair.
        X_train,X_test,y_train,y_test = train_test_split(X,  y, test_size = 50, random_state = SEED_INT, stratify=y)

        selector = SelectKBest(chi2, k=i)
        # Only train data is used to fit the selectKbest selector.
        selector.fit(X_train, y_train)
        # However, the X_test data still makes use of it.
        # This is to simulate a simple case of testing the model with new data that can not influence the selector.
        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)

        return X_train,X_test,y_train,y_test

    def test_features(self,i, classifier_id, X, y ):
        """Train and test a model after preprocessing:"""
        # https://blog.finxter.com/python-list-copy/
        # Copy is atomic and rest of the variables are thread safe in the current namespace.
        # This also re-intitializes the data in the local scope. 
        i = int(i)
        X = X.copy()
        y = y.copy()    
        X_train,X_test,y_train,y_test = self.preprocess(i,X, y)   

        # Model selector:
        # The negation of the real accuracy is returned from test_features because the fminbound optimizer that uses this function...
        # tries to find the minimum by default.
        # The optimizers output value can be negated again upon termination of the optimizer to give the real accuracy.
        if("dec_tree_model" == classifier_id):
            return -dec_tree_model(X_train,X_test,y_train,y_test)
        if("log_reg_model" == classifier_id):
            return -log_reg_model(X_train,X_test,y_train,y_test)
        if("rand_forest_model" == classifier_id):
            return -rand_forest_model(X_train,X_test,y_train,y_test)      
        if("naive_bays_model" == classifier_id):
            return -naive_bays_model(X_train,X_test,y_train,y_test)
        


def main():

    start_time = time.time()
    f = open('tf_matrix.csv', 'r', encoding="utf-8")
    data = pd.read_csv(f, header=0) 

    # Personality pairs used for outputs:
    pairs  =  ["_I_E_","_N_S_", "_T_F_", "_J_P_"] 

    # Features are all word/tokens that occur in at least one of the users posts (based on "word_bank" in the transfrom.py programm).
    features = list(data.keys())
    features = features[:-5]

    # X is the tf matrix from tf_matrix.csv (not the personality data)
    X = data[features]
    X = X.values

    # convert to X csr matrix becuase the numpy array is sparse
    X = csr_matrix(X)

    classification_tests = tests()



    # The eventual shape of best_in_class is [4][2][4].
    # The first dimension is the personality pair being tested.
    # The seconds dimension is two lists.
    # One is a list of accuracies scores for each model type,
    # The other is the by index correspnding optimal number of features that produce the accuracy scores for each model type. 


    # "best_in_class" is used to theoreticly compute the accuracy of any single classifer type predicting the full personality (4 types correctly)
    # "best_in_class" is also used to theoreticlly compute the accuracy when the absolute best optmized classfier type...
    # is used to predict each personailty pair (4 types correctly)
    best_in_class = []


    print("Training and testing with (39 X 16) = 624 users for each personality pair\n")


    for item in pairs:  
        # Target values of the current personality pair:
        y = list(data[item])
        # Each model is tested for each pair prediction.
        local_optimas_dec  = []
        local_optimas_reg  = []
        local_optimas_for  = []
        local_optimas_bays = []   
        # "i" and "gap_size" determine the bounds of the fminbound opimization on the test_features function used below.
        i = 0
        gap_size = 8
        print(item, "Classification:")
        while(i<6):           
            # The optimiztion function fminbound finds the best value between the bounds of the number of features ((i)*gap_size+1 to (i+1)*gap_size+1)...
            # that maximizes model perfrormance.
            # Multithreading is used here for potential improvements in runtime.  
            t1 = return_thread(group=None,target=fminbound,
                            kwargs={"func" : classification_tests.test_features,"x1" : (i)*gap_size+1, "x2" : (i+1)*gap_size+1,"args": ("dec_tree_model",X, y), "full_output" : True, "disp" :0})
            t2 = return_thread(group=None,target=fminbound,
                            kwargs={"func" : classification_tests.test_features,"x1" : (i)*gap_size+1, "x2" : (i+1)*gap_size+1,"args": ("log_reg_model",X, y),"full_output" : True, "disp" :0})
            t3 = return_thread(group=None,target=fminbound,
                            kwargs={"func" : classification_tests.test_features,"x1" : (i)*gap_size+1, "x2" : (i+1)*gap_size+1,"args": ("rand_forest_model",X, y),"full_output" : True, "disp" :0})
            t4 = return_thread(group=None,target=fminbound,
                            kwargs={"func" : classification_tests.test_features,"x1" : (i)*gap_size+1, "x2" : (i+1)*gap_size+1,"args": ("naive_bays_model",X, y),"full_output" : True, "disp" :0})
            t1.start()
            t2.start()
            t3.start()
            t4.start()  

            local_optima_dec = t1.join()
            local_optima_reg = t2.join()
            local_optima_for = t3.join()
            local_optima_bays = t4.join()   
            
            
            # The negation is returned (see "test_features" function)
            local_optimas_dec.append((-local_optima_dec[1], int(local_optima_dec[0])))
            local_optimas_reg.append((-local_optima_reg[1], int(local_optima_reg[0])))
            local_optimas_for.append((-local_optima_for[1], int(local_optima_for[0])))
            local_optimas_bays.append((-local_optima_bays[1], int(local_optima_bays[0])))  


            # -local_optima_dec[1] is an accuracy score for the best local (within a range of number of features) model for decision tree.
            # int(local_optima_dec[0]) is number of features for the best local(within a range of number of features) model for decision tree.
            # A similair case is true for the other classfiers.
            # This is used to show progress:
            print("Iteration:", i+1)
            print("Decision Tree Test:",  "Accuracy:",-local_optima_dec[1], "Features:", int(local_optima_dec[0]))                       
            print("Logistic Regression Test:",  "Accuracy:",-local_optima_reg[1], "Features:", int(local_optima_reg[0]))       
            print("Random Forest Test:", "Accuracy:",-local_optima_for[1], "Features:", int(local_optima_for[0]))  
            print("Naive Bays Test:",  "Accuracy:",-local_optima_bays[1], "Features:", int(local_optima_bays[0]),"\n")  
            
            i += 1

        # Find the model with the most optimal number of features for each classfier for the given pair.
        # The optimal model has the highest accuracy for the classfier in question.
        optimal_dec = max(local_optimas_dec, key = lambda x: x[0])
        optimal_reg = max(local_optimas_reg, key = lambda x: x[0])
        optimal_for = max(local_optimas_for, key = lambda x: x[0])
        optimal_bays = max(local_optimas_bays, key = lambda x: x[0])
        

        # Print the accuracy and the number of features for the best of each classifier for the given pair.
        print(item, "Classification:")
        print("Best Predictor Function Scores:")
        print("Decision Tree:","Features:",optimal_dec[1],"Accuracy:",optimal_dec[0])
        print("Logistic Regession:","Features:",optimal_reg[1],"Accuracy:",optimal_reg[0])
        print("Random Forest:", "Features:",optimal_for[1],"Accuracy:",optimal_for[0])
        print("Naive Bays:", "Features:",optimal_bays[1],"Accuracy:",optimal_bays[0],"\n")
        

        best_in_class.append([[optimal_dec[0], optimal_reg[0], optimal_for[0],optimal_bays[0]],
                            [optimal_dec[1], optimal_reg[1], optimal_for[1], optimal_bays[1]]])


    # Classifier types used for outputs:
    classifiers = ["Decision Tree", "Logistic Regression", "Random Forest", "Naive Bays"]

    print("Complete Myers Briggs Prediction for individual Classifiers:\n")

    f = open("results.txt","w")

    f.write("Complete Myers Briggs Prediction for individual Classifiers:")
    f.write("\n\n")

    # Find the best models that uses the same classifer type for each personality pair and output results.
    for i in range (0,4):
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
    best_classifer_and_nof_features_per_personailty_pair = []
    best_model_accuracy  = 1

    i = 0
    for pair in best_in_class:
        classifier_acc_nof_features = list(zip(classifiers, pair[0], pair[1]))
        classifier_max_acc_nof_features = max(classifier_acc_nof_features, key = (lambda x: x[1]))
        best_model_accuracy*= classifier_max_acc_nof_features[1]
        best_classifer_and_nof_features_per_personailty_pair.append("Pair: "+str(pairs[i])+\
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

    for item in best_classifer_and_nof_features_per_personailty_pair:
        f.write(item)
        f.write("\n")
        print(item)
    f.close()


    # Computation time:
    print("Done")
    print("Full compute time:",float((time.time() - start_time)/60), "Minutes")



if __name__ == "__main__":
    main()


