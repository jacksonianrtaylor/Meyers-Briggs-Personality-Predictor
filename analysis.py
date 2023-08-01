
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



#Inherits from the Thread class so that a version can be created that...
#returns function parameters return value in the join method
class return_thread(Thread):
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

#classification models
#they all return accuracy scores
#which is simply the number of correct predictions divided by the number of predictions
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

def fit_tree(X_train,X_test,y_train,y_test, alpha):
    if(alpha < 0):
        alpha = 0
    dec_tree_a = DecisionTreeClassifier(random_state = rseed, ccp_alpha = alpha)
    dec_tree_a.fit(X_train,y_train)
    y_pred = dec_tree_a.predict(X_test)   
    return accuracy_score(y_test, y_pred)
    
def dec_tree_model(X_train,X_test,y_train,y_test):
    dec_tree = DecisionTreeClassifier(random_state = rseed)
    path = dec_tree.cost_complexity_pruning_path(X_train, y_train)
    alphas = path['ccp_alphas']   
    scores  = []
    for alpha in alphas:
        scores.append(fit_tree(X_train,X_test,y_train,y_test, alpha))             
    scores.sort(reverse  = True)   
    return scores[0]    
    

#tests class
#this is what is used to test the accuracy of models for varied number of feature inputs...
#that are selected with chi-squared feature selection
#the input features are simply word counts
class tests():   
    def __init__(self):
        #This is used for consistent train test splits for a given runtime for all models
        #however, it has been replaced with a constant seed...
        #to produce simlair consistencey over runtimes
        #see rseed variable above
        self.seed = random.randint(0, 42) 

    def preprocess(self, i,X, y):
        #note: the current configuration uses only the train data to fit the selectKbest selector
        #This is to simulate a simple case of testing the model with new data
        #the test data can use the "selector" but should not contribute to fiting it
        X_train,X_test,y_train,y_test = train_test_split(X,  y, test_size = 50, random_state = rseed, stratify=y)
        #LOOK, NEW: the test and train data should be split 50/50 for equal output values (0 and 1)
        #this could be acheived using two train test splits and combining them
        #

        selector = SelectKBest(chi2, k=i)
        selector.fit(X_train, y_train)
        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)

        return X_train,X_test,y_train,y_test

    def test_features(self,i, model_id, X, y ):
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

#load data from tf_matrix.csv that was populated by the transform.py program
time_t = time.time()
f = open('tf_matrix.csv', 'r', encoding="utf-8")
data = pd.read_csv(f, header=0) 

#used for outputs
pairs  =  ["_I_E_","_N_S_", "_T_F_", "_J_P_"] 

#features are all word/tokens that occur in at least one text
features = list(data.keys())
features = features[:-5]

#X is all the occurances of the words for each user
X = data[features]
X = X.values

#convert to X csr matrix becuase the numpy array is sparse
X = csr_matrix(X)

classification_tests = tests()

#list of the accuracies of the best models of each type (Decision Tree, Logistic Regession, Random Forest, Naive Bays)....
#for predicting each personality pair
Best_in_class = []

#used to output to the results.csv file
out_df = pd.DataFrame()

#analysis
print("Testing 39 users for each of the 16 meyers briggs personalities...\n")
c = 0
for item in pairs:  
    c = c+1
    #not necessary to use list(data[item]) here...
    y = data[item]
    #each model is tested for each pair prediction
    local_optimas_dec  = []
    local_optimas_reg  = []
    local_optimas_for  = []
    local_optimas_bays = []   
    #i and gap_size determine the bounds of the fminbound opimization on the test_features function
    i = 0
    gap_size = 8
    print(item, "Classification:")
    while(i<6):           
        #optimiztion function fminbound finds the best value between the bounds of the number of features...
        #that minimizes test features

        #multithreading is used for small to medium improvements in runtime  

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

        local_optima_1 = t1.join()
        local_optima_2 = t2.join()
        local_optima_3 = t3.join()
        local_optima_4 = t4.join()   
        
        # slower and without the use of multithreading
        # local_optima_1 = fminbound(classification_tests.test_features,x1 = (i)*gap_size+1, x2 =(i+1)*gap_size+1 , args = ("dec_tree_model",X, y), full_output  = True, disp   =0)
        # local_optima_2 = fminbound(classification_tests.test_features,x1 = (i)*gap_size+1, x2 =(i+1)*gap_size+1 , args = ("log_reg_model",X, y), full_output  = True, disp   =0)
        # local_optima_3 = fminbound(classification_tests.test_features,x1 = (i)*gap_size+1, x2 =(i+1)*gap_size+1 , args = ("rand_forest_model",X, y), full_output  = True, disp   =0)
        # local_optima_4 = fminbound(classification_tests.test_features,x1 = (i)*gap_size+1, x2 =(i+1)*gap_size+1 , args = ("naive_bays_model",X, y), full_output  = True, disp   =0)
        
        #the negation is returned (see test_features function)
        local_optimas_dec.append((-local_optima_1[1], int(local_optima_1[0])))
        local_optimas_reg.append((-local_optima_2[1], int(local_optima_2[0])))
        local_optimas_for.append((-local_optima_3[1], int(local_optima_3[0])))
        local_optimas_bays.append((-local_optima_4[1], int(local_optima_4[0])))  


        #local_optima_1[1] is a score
        #and local_optima_1[0] is number of features used that gave the accuracy score 
        #this is to show progress but may be out of order with use of multithreading

        print("Iteration:", i+1)
        print("Decision Tree Test:",  "Score:",-local_optima_1[1], "Features:", int(local_optima_1[0]))                       
        print("Logistic Regression Test:",  "Score:",-local_optima_2[1], "Features:", int(local_optima_2[0]))        
        print("Random Forest Test:", "Score:",-local_optima_3[1], "Features:", int(local_optima_3[0]))       
        print("Naive Bays Test:",  "Score:",-local_optima_4[1], "Features:", int(local_optima_4[0]),"\n")   
        
        i += 1

    #find the best accuracy for the best number of features for each model for a certain personality pair
    local_optimas_dec.sort(reverse  = True) 
    local_optimas_reg.sort(reverse  = True)
    local_optimas_for.sort(reverse  = True)
    local_optimas_bays.sort(reverse  = True)
    
    #outputs to be used later in results.csv
    out_df[item] = ["Decision Tree: "+"Features: "+str(local_optimas_dec[0][1])+" Accuracy: "+str(local_optimas_dec[0][0])
    ,"Logistic Regession: "+"Features: "+str(local_optimas_reg[0][1])+" Accuracy: "+str(local_optimas_reg[0][0])
    ,"Random Forest: "+ "Features: "+str(local_optimas_for[0][1])+" Accuracy: "+str(local_optimas_for[0][0])
    ,"Naive Bays: "+ "Features: "+str(local_optimas_bays[0][1])+" Accuracy: "+str(local_optimas_bays[0][0])]
    

    #print results directly above
    print(item, "Classification:")
    print("Best Predictor Function Scores:")
    print("Decision Tree:","Features:",local_optimas_dec[0][1],"Accuracy:",local_optimas_dec[0][0])
    print("Logistic Regession:","Features:",local_optimas_reg[0][1],"Accuracy:",local_optimas_reg[0][0])
    print("Random Forest:", "Features:",local_optimas_for[0][1],"Accuracy:",local_optimas_for[0][0])
    print("Naive Bays:", "Features:",local_optimas_bays[0][1],"Accuracy:",local_optimas_bays[0][0],"\n")
    
    #Best of class is a list of optimal scores for each model for each personality pair
    #note: This append is just for a single personality pair
    #note: local_optimas_dec[0][0] gives the best decision tree score since local_optimas_dec is sorted backwards
    Best_in_class.append([local_optimas_dec[0][0], 
                          local_optimas_reg[0][0], 
                          local_optimas_for[0][0], 
                          local_optimas_bays[0][0]])



#used for outputs
results = []
classifiers = ["Decision Tree: ", "Logistic Regression: ", "Random Forest: ", "Naive Bays: "]

#header 1
print("Complete Myers Briggs Prediction for individual Classifiers:")

#find the accuracy score for each pair for each model
for x in range (0,4):
    #product is the score for a models performance in prediting all personality pairs correctly
    #it is a theoretical score using law of independence
    product =1
    for item in Best_in_class:
        # item controls the personailty pair in question
        # x controls the model
        product*= item[x]
    print(classifiers[x], product)
    results.append(str(classifiers[x])+ str(product))

#header 2
print("\nMyers Briggs Prediction from the best of each Classifier:")

#find the absolute best model
#product is used in the same way as above
product  = 1
for item in Best_in_class:
    item.sort(reverse  = True)
    product*= item[0]


#this shows the score by using a model consistently for all the pair predictions
out_df["Best in Class:"] = results

#this ouputs the theoretical score using the best model for each pair prediction
print(product)
out_df["Overall Best"] = [str(product), "", "","" ]

#outputs
out_file = open('results.csv', 'w', encoding="utf-8")
out_df.to_csv(out_file,index  = False)
out_file.close()   

#time to compute
print("Full compute time:",time.time()-time_t,"Seconds")


#Note: it is important to state in the readme that one personailty pair prediction should not have an impact on 
#another personality pair since there is the same number of people for each of the 16 personalilties
#This eliminates bais for common vs uncommon personalities
#it means that a full prediction is not made but is theoreticaly accurate based on the predictions of each pair
#using the law of independence and the best scored model for each personality pair...








