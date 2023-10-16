
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


#some normalization techniques were tested that decreased the time for model convergence but produced worse results for accurcay scores
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


#global random seed for consistent train test splits 
#and randomstates for certain models
rseed = 10


class return_thread(Thread):
    """Thread class that returns a value in the join method"""
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

#classification models:
#they all return accuracy scores (the number of correct predictions divided by the number of predictions)

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
    rand_for = RandomForestClassifier(random_state = rseed)
    rand_for.fit(X_train,y_train)
    y_pred=rand_for.predict(X_test)   
    return accuracy_score(y_test, y_pred)

def dec_tree_model(X_train,X_test,y_train,y_test):
    """The decision tree model implements cost complexity pruning"""
    dec_tree = DecisionTreeClassifier(random_state = rseed)
    path = dec_tree.cost_complexity_pruning_path(X_train, y_train)
    alphas = path['ccp_alphas']   
    scores  = []
    for alpha in alphas:
        scores.append(fit_tree(X_train,X_test,y_train,y_test, alpha))             
    scores.sort(reverse  = True)   
    return scores[0]   

def fit_tree(X_train,X_test,y_train,y_test, alpha):
    "get the accuracy score of a specific tree with given ccp_alpha"
    if(alpha < 0):
        alpha = 0
    dec_tree_a = DecisionTreeClassifier(random_state = rseed, ccp_alpha = alpha)
    dec_tree_a.fit(X_train,y_train)
    y_pred = dec_tree_a.predict(X_test)   
    return accuracy_score(y_test, y_pred)
    
   


class tests():   
    """
    This is what is used to test the accuracy of models for varied number of feature inputs
    that are selected with chi-squared feature selection
    The input features are simply word counts.
    """
    def __init__(self):
        #This is used for consistent train test splits for a given runtime for all models
        #however, it has been replaced with a constant seed to produce simlair consistencey over runtimes
        #rseed is used instead
        self.seed = random.randint(0, 42) 

    def preprocess(self, i,X, y):
        """Make the data ready for the model"""
        #stratefy makes sure the same proportions of target values in the full dataset is preseved in the
        #train and test data both the X_test and y_test are split 50/50
        X_train,X_test,y_train,y_test = train_test_split(X,  y, test_size = 50, random_state = rseed, stratify=y)

        selector = SelectKBest(chi2, k=i)
        #only train data is used to fit the selectKbest selector...
        selector.fit(X_train, y_train)
        #but the X_test data still makes use of it
        #This is to simulate a simple case of testing the model with new data that can not influence the selector
        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)

        return X_train,X_test,y_train,y_test

    def test_features(self,i, model_id, X, y ):
        """Train and test a model after preprocessing"""
        #https://blog.finxter.com/python-list-copy/
        #copy is atomic and rest of the variables are thread safe in the current namespace
        #this also re-intitializes the data in the local scope
        i = int(i)
        X = X.copy()
        y = y.copy()    
        X_train,X_test,y_train,y_test = self.preprocess(i,X, y)   

        #model selector
        #the negation of the real accuracy is returned from test_features because the fminbound optmizer that uses this function...
        #tries to find the minimum by default.
        #the optimizers output value can be negated again upon termination of the optimizer to give the real accuracy

        if("dec_tree_model" == model_id):
            return -dec_tree_model(X_train,X_test,y_train,y_test)
        if("log_reg_model" == model_id):
            return -log_reg_model(X_train,X_test,y_train,y_test)
        if("rand_forest_model" == model_id):
            return -rand_forest_model(X_train,X_test,y_train,y_test)      
        if("naive_bays_model" == model_id):
            return -naive_bays_model(X_train,X_test,y_train,y_test)
        

#main

time_t = time.time()
f = open('tf_matrix.csv', 'r', encoding="utf-8")
data = pd.read_csv(f, header=0) 

#pairs used for outputs
pairs  =  ["_I_E_","_N_S_", "_T_F_", "_J_P_"] 

#features are all word/tokens that occur in at least one (from word bank in the transfrom.py programm)
features = list(data.keys())
features = features[:-5]

#X is all the occurances of the words for each user the tf mtrix from tf_mtrix.csv (not the personality data)
X = data[features]
X = X.values

#convert to X csr matrix becuase the numpy array is sparse
X = csr_matrix(X)

classification_tests = tests()


#this is what it is:
#a list of the accuracy scores for the best model of each model type (Decision Tree, Logistic Regession, Random Forest, Naive Bays) for every personalitiy pair
#this list becomes a (4 personality pairs X 4 model types) = 16 model accuracies


#this is how it is used:
#this is used to multiply the accuracy scores of the same model type for each personailty pair to aproximate the comeplete personailty accuracy score
#for a given model. This means there will be 4 accuracy scores (one for eahc model)
#this is also used to find the absolute best model (take the best models from the Best in class for each personality pair)
Best_in_class = []

#used to output to the results.csv file to observe outpus in a different way
#note: may need to change later
#LOOK
#this is being replaced
# out_df = pd.DataFrame()

print("Training and testing with (39 X 16) = 624 users for each personality pair\n")


for item in pairs:  
    #target values of the current personality pair
    y = list(data[item])
    #each model is tested for each pair prediction
    local_optimas_dec  = []
    local_optimas_reg  = []
    local_optimas_for  = []
    local_optimas_bays = []   
    #i and gap_size determine the bounds of the fminbound opimization on the test_features function used below
    i = 0
    gap_size = 8
    print(item, "Classification:")
    while(i<6):           
        #optimiztion function fminbound finds the best value between the bounds of the number of features ((i)*gap_size+1 to (i+1)*gap_size+1)...
        #that maximizes model perfromance
        #multithreading is used here for small to medium improvements in runtime  
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
        
        # slower version of the above without the use of multithreading
        # local_optima_1 = fminbound(classification_tests.test_features,x1 = (i)*gap_size+1, x2 =(i+1)*gap_size+1 , args = ("dec_tree_model",X, y), full_output  = True, disp   =0)
        # local_optima_2 = fminbound(classification_tests.test_features,x1 = (i)*gap_size+1, x2 =(i+1)*gap_size+1 , args = ("log_reg_model",X, y), full_output  = True, disp   =0)
        # local_optima_3 = fminbound(classification_tests.test_features,x1 = (i)*gap_size+1, x2 =(i+1)*gap_size+1 , args = ("rand_forest_model",X, y), full_output  = True, disp   =0)
        # local_optima_4 = fminbound(classification_tests.test_features,x1 = (i)*gap_size+1, x2 =(i+1)*gap_size+1 , args = ("naive_bays_model",X, y), full_output  = True, disp   =0)
        
        #the negation is returned (see test_features function)
        local_optimas_dec.append((-local_optima_dec[1], int(local_optima_dec[0])))
        local_optimas_reg.append((-local_optima_reg[1], int(local_optima_reg[0])))
        local_optimas_for.append((-local_optima_for[1], int(local_optima_for[0])))
        local_optimas_bays.append((-local_optima_bays[1], int(local_optima_bays[0])))  


        #-local_optima_dec[1] is an accuracy score for the best local(within a range of number of features) model for decision tree
        #int(local_optima_dec[0]) is number of features for the best local(within a range of number of features) model for decision tree
        #similair is true for the other classfiers
        #this is used to show progress 
        print("Iteration:", i+1)
        print("Decision Tree Test:",  "Accuracy:",-local_optima_dec[1], "Features:", int(local_optima_dec[0]))                       
        print("Logistic Regression Test:",  "Accuracy:",-local_optima_reg[1], "Features:", int(local_optima_reg[0]))       
        print("Random Forest Test:", "Accuracy:",-local_optima_for[1], "Features:", int(local_optima_for[0]))  
        print("Naive Bays Test:",  "Accuracy:",-local_optima_bays[1], "Features:", int(local_optima_bays[0]),"\n")  
        
        i += 1

    #find the model with the most optimal number of features for each classfier for the the given pair
    #the optimal model has the highest accuracy for the classfier in question
    optimal_dec = max(local_optimas_dec, key = lambda x: x[0])
    optimal_reg = max(local_optimas_reg, key = lambda x: x[0])
    optimal_for = max(local_optimas_for, key = lambda x: x[0])
    optimal_bays = max(local_optimas_bays, key = lambda x: x[0])
    

    #print the accuracy and the number of features for the best of each classifier for the given pair
    print(item, "Classification:")
    print("Best Predictor Function Scores:")
    print("Decision Tree:","Features:",optimal_dec[1],"Accuracy:",optimal_dec[0])
    print("Logistic Regession:","Features:",optimal_reg[1],"Accuracy:",optimal_reg[0])
    print("Random Forest:", "Features:",optimal_for[1],"Accuracy:",optimal_for[0])
    print("Naive Bays:", "Features:",optimal_bays[1],"Accuracy:",optimal_bays[0],"\n")
    

    #Best_in_class is a list of pairs that include four accuracy scores for each of the four classifers on the personailty type...
    #and four number of optimal features corresponding to these accuracy scores
    #each accuracy score is the best accuracy score for the coresponding classifer for the given pair
    #each of the optimal number of features is the inputs to the model that produces this accuracy score
    Best_in_class.append([[optimal_dec[0],
                          optimal_reg[0],
                          optimal_for[0], 
                          optimal_bays[0]],
                          [optimal_dec[1],
                          optimal_reg[1],
                          optimal_for[1], 
                          optimal_bays[1]]])


#used for outputs
classifiers = ["Decision Tree", "Logistic Regression", "Random Forest", "Naive Bays"]

print("Complete Myers Briggs Prediction for individual Classifiers:")

f = open("results.txt","w")

#find the best models that uses the same classifer type for each personality pair and out put them
for i in range (0,4):
    pair_wise_features = ""
    model_accuracy =1
    j = 0
    for pair in Best_in_class:
        model_accuracy*= pair[0][i]
        pair_wise_features+= "Pair: "+pairs[j]+" Features: "+str(pair[1][i])
        j+=1
        
    print("Classifier:", classifiers[i])
    print("Accuracy:",str(model_accuracy))
    print(pair_wise_features)
    f.write("Classifier: "+classifiers[i])
    f.write("\n")
    f.write("Accuracy: "+str(model_accuracy))
    f.write("\n")
    f.write(pair_wise_features)
    f.write("\n")



#find the absolute best model...
best_classifer_and_number_features_per_personailty_pair = ""
best_model_accuracy  = 1

i = 0
for pair in Best_in_class:
    acc_with_model_type = list(zip(pair[0], pair[1], classifiers))
    acc_with_model_type = sorted(acc_with_model_type, key = (lambda x: x[0]) ,reverse = True)
    best_model_accuracy*= acc_with_model_type[0][0]
    best_classifer_and_number_features_per_personailty_pair+=("Pair: "+str(pairs[i])+ " Classifier: "+str(acc_with_model_type[0][2])+" Number of features: "+str(acc_with_model_type[0][1])+" ")
    i+=1


#Print and output the best model accuracy and the top models for each personality pair
f.write("Myers Briggs Prediction from the best of each Classifier:")
f.write("\n")
f.write("Best model accuracy: " + str(best_model_accuracy))
f.write("\n")
f.write(best_classifer_and_number_features_per_personailty_pair)
f.close()
print("Myers Briggs Prediction from the best of each Classifier:")
print("Best model accuracy:", best_model_accuracy)
print(best_classifer_and_number_features_per_personailty_pair)
  


#computation time
print("Done")
print("Full compute time:",float((time.time() - time_t)/60), "Minutes")






#note: need to implement main!!!

#last output

# Complete Myers Briggs Prediction for individual Classifiers:
# Decision Tree:  0.41827968
# Logistic Regression:  0.4061952
# Random Forest:  0.440832
# Naive Bays:  0.3005184

# Myers Briggs Prediction from the best of each Classifier:
# 0.4738944
# Full compute time: 414.5868921279907 Seconds



    # local_optimas_dec.sort(reverse  = True) 
    # local_optimas_reg.sort(reverse  = True)
    # local_optimas_for.sort(reverse  = True)
    # local_optimas_bays.sort(reverse  = True)


    # #print the accuracy and the number of features for each of the classifiers for the given pair
    # print(item, "Classification:")
    # print("Best Predictor Function Scores:")
    # print("Decision Tree:","Features:",local_optimas_dec[0][1],"Accuracy:",local_optimas_dec[0][0])
    # print("Logistic Regession:","Features:",local_optimas_reg[0][1],"Accuracy:",local_optimas_reg[0][0])
    # print("Random Forest:", "Features:",local_optimas_for[0][1],"Accuracy:",local_optimas_for[0][0])
    # print("Naive Bays:", "Features:",local_optimas_bays[0][1],"Accuracy:",local_optimas_bays[0][0],"\n")
    

    # #Best of class is a list of pairs that include four accuracy scores for each of the classifers
    # #and optimal number of features for used for each of the four classifiers
    # Best_in_class.append([[local_optimas_dec[0][0],
    #                       local_optimas_reg[0][0],
    #                       local_optimas_for[0][0], 
    #                       local_optimas_bays[0][0]],
    #                       [local_optimas_dec[0][1],
    #                       local_optimas_reg[0][1],
    #                       local_optimas_for[0][1], 
    #                       local_optimas_bays[0][1]]])