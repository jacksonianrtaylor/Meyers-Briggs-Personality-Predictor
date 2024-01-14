import pandas as pd
import time
import copy
import random
from threading import Thread
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


# Global random seed: 
# Original:
# SEED_INT = 10 (.459)
# SEED_INT = 5 (.428)
# SEED_INT = 15 (0.408)
# SEED_INT = 20 (0.486)


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
    #LOOK: remove entropy if worse
    dec_tree = DecisionTreeClassifier(criterion = "entropy", random_state = SEED_INT)
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

    def preprocess(self, i, X_train,X_test,y_train,y_test):
        """Transform the data to be ready for a model:"""
        #  Stratify makes sure the same proportions of target values in the full dataset is preseved in the train and test data 
        #  Both the y_train and y_test are split 50/50 between any given pair.

        selector = SelectKBest(chi2, k=i)
        # Only train data is used to fit the selectKbest selector.
        # This is to simulate a simple case of testing the model with new data that can not influence the selector.
        selector.fit(X_train, y_train)
        # However, the X_test data still makes use of it.
        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)

        return X_train,X_test,y_train,y_test


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


    def test_features(self,i, classifier_id, X, y):
        """Train and test a model after preprocessing:"""
        num_features = round(i[0])

        # X_train,X_test,y_train,y_test = self.preprocess(num_features, X, y)   
        #Note: data is already shuffled with preprocess...

        #LOOK: There is a method in which fold the of features are selected again acording to the data at hand
        #but I hope the methods present is enough to suffice


        X_new = X.toarray().tolist()
        y_new = y

        temp = list(zip(X_new, y_new))
        random.shuffle(temp)
        X_new, y_new = zip(*temp)

        # idea: split the data by the target value 
        # shuffle list of pairs for one type of target values
        # shuffle list of pairs for other type of target values
        # select 25 from each list 
        # ....


        # Model selector:
        # The negation of the real accuracy is returned from test_features because the differential_evolution optimizer that uses this function...
        # tries to find the minimum by default.
        # The optimizers output value can be negated again upon termination of the optimizer to give the real accuracy.
        acc_sum = 0
        for i in range(5):


            X_test = X_new[50*i:50*(i+1)]
            y_test = y_new[50*i:50*(i+1)]
            if(i==0):
                X_train = X_new[50*(i+1):]
                y_train = y_new[50*(i+1):]
            else: 
                X_train = X_new[:50*i] + X_new[50*(i+1):]
                y_train = y_new[:50*i] + y_new[50*(i+1):]

            #LOOK: need to statify the users: same numebr of each personality is in train and test 

            X_train,X_test,y_train,y_test = self.preprocess(num_features,X_train,X_test,y_train,y_test)

            if("dec_tree_model" == classifier_id):
                acc_sum+=-dec_tree_model(X_train,X_test,y_train,y_test)
            if("log_reg_model" == classifier_id):
                acc_sum+=-log_reg_model(X_train,X_test,y_train,y_test)
            if("rand_forest_model" == classifier_id):
                acc_sum+=-rand_forest_model(X_train,X_test,y_train,y_test) 
            if("naive_bays_model" == classifier_id):
                acc_sum+=-naive_bays_model(X_train,X_test,y_train,y_test)


        return acc_sum/5
        

        


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

    classification_tests = tests()


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
        # Target values of the current personality pair:
        y = list(data[item])
          
        # The optimiztion function, differential_evolution, finds the best value between the bounds of the number of features that maximizes model performance.
        # Multithreading is used here for some improvements in runtime.

        #LOOK: try multiples runs without seeding the optimizer  
        #does
        t1 = return_thread(group=None,target=differential_evolution,
                        kwargs={"func" : classification_tests.test_features,"bounds" : [(30,50)],
                                 "args": ("dec_tree_model",copy.deepcopy(X), copy.deepcopy(y)), "seed": SEED_INT})
        t2 = return_thread(group=None,target=differential_evolution,
                        kwargs={"func" : classification_tests.test_features,"bounds" :  [(30,50)],
                                 "args": ("log_reg_model",copy.deepcopy(X), copy.deepcopy(y)), "seed": SEED_INT})
        t3 = return_thread(group=None,target=differential_evolution,
                        kwargs={"func" : classification_tests.test_features,"bounds" :  [(30,50)],
                                 "args": ("rand_forest_model",copy.deepcopy(X), copy.deepcopy(y)), "seed": SEED_INT})
        t4 = return_thread(group=None,target=differential_evolution,
                        kwargs={"func" : classification_tests.test_features,"bounds" :  [(30,50)],
                                 "args": ("naive_bays_model",copy.deepcopy(X), copy.deepcopy(y)), "seed": SEED_INT})
        
        # t1 = return_thread(group=None,target=differential_evolution,
        #                 kwargs={"func" : classification_tests.test_features,"bounds" : [(30,50)],
        #                          "args": ("dec_tree_model",copy.deepcopy(X), copy.deepcopy(y))})
        # t2 = return_thread(group=None,target=differential_evolution,
        #                 kwargs={"func" : classification_tests.test_features,"bounds" :  [(30,50)],
        #                          "args": ("log_reg_model",copy.deepcopy(X), copy.deepcopy(y))})
        # t3 = return_thread(group=None,target=differential_evolution,
        #                 kwargs={"func" : classification_tests.test_features,"bounds" :  [(30,50)],
        #                          "args": ("rand_forest_model",copy.deepcopy(X), copy.deepcopy(y))})
        # t4 = return_thread(group=None,target=differential_evolution,
        #                 kwargs={"func" : classification_tests.test_features,"bounds" :  [(30,50)],
        #                          "args": ("naive_bays_model",copy.deepcopy(X), copy.deepcopy(y))})
        t1.start()
        t2.start()
        t3.start()
        t4.start()  

        optimal_dec_result = t1.join()
        optimal_reg_result = t2.join()
        optimal_for_result = t3.join()
        optimal_bays_result = t4.join()   
            

        # Print the accuracy and the number of features for the best of each classifier for the given pair.
        print(item, "Classification:")
        print("Best Predictor Function Scores:")
        print("Decision Tree:","Accuracy:", -optimal_dec_result.fun,"Features:",int(optimal_dec_result.x[0]))
        print("Logistic Regession:","Accuracy:",-optimal_reg_result.fun,"Features:",int(optimal_reg_result.x[0]))
        print("Random Forest:", "Accuracy:",-optimal_for_result.fun,"Features:",int(optimal_for_result.x[0]))
        print("Naive Bays:", "Accuracy:",-optimal_bays_result.fun,"Features:",int(optimal_bays_result.x[0]),"\n")
        

        best_in_class.append([[-optimal_dec_result.fun, -optimal_reg_result.fun,
                            -optimal_for_result.fun,-optimal_bays_result.fun],
                            [int(optimal_dec_result.x[0]), int(optimal_reg_result.x[0]),
                            int(optimal_for_result.x[0]), int(optimal_bays_result.x[0])]])


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


