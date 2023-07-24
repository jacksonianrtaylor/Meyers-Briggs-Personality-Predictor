#Jackson Taylor
#Student ID: 00001590147
#CSCI 182: Web and Data Mining (35622)
import pandas as pd
import scipy
import sklearn
import time
import random
import threading
from threading import Thread
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.optimize import fminbound
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import normalize

#Inherits from the Thread class so that a version can be created that returns function parameters return value in the join method
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
#they all return the accuracy score
#which is simply the number of correct predictions over the number of predictions
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
    rand_for = RandomForestClassifier()
    rand_for.fit(X_train,y_train)
    y_pred=rand_for.predict(X_test)   
    return accuracy_score(y_test, y_pred)

def fit_tree(X_train,X_test,y_train,y_test, alpha):
    if(alpha < 0):
        alpha = 0
    dec_tree_a = DecisionTreeClassifier(ccp_alpha = alpha)
    dec_tree_a.fit(X_train,y_train)
    y_pred = dec_tree_a.predict(X_test)   
    return accuracy_score(y_test, y_pred)
    
def dec_tree_model(X_train,X_test,y_train,y_test):
    dec_tree = DecisionTreeClassifier()
    path = dec_tree.cost_complexity_pruning_path(X_train, y_train)
    alphas = path['ccp_alphas']   
    scores  = []
    for alpha in alphas:
        scores.append(fit_tree(X_train,X_test,y_train,y_test, alpha))             
    scores.sort(reverse  = True)   
    return scores[0]    
    

#test class
class tests():   
    def __init__(self):
        #seed remains the same for all test cases in order to make comparisons to multiple classifiers
        #the seed is still random so the same outcome will not happen everytime
        self.seed = random.randint(0, 42) 

    def preprocess(self, i,X, y):
        #chi squared feature selection and returns the X_train,X_test,y_train,y_test
        #i is the number of feature to use
        X_new = SelectKBest(chi2, k=i).fit_transform(X, y)
        return train_test_split(X_new,  y,test_size = 100, random_state = self.seed)

    def test_features(self,i, model_id,X, y ):
        #https://blog.finxter.com/python-list-copy/
        #copy is atomic and rest of the variables are thread safe in the current namespace
        #this also re-intitializes the data in the local scope
        i = int(i)
        X = X.copy()
        y = y.copy()    
        X_train,X_test,y_train,y_test = self.preprocess(i,X, y)   

        #perform model selection  
        #why is the negation returned???
        if("dec_tree_model" == model_id):
            return -dec_tree_model(csc_matrix(X_train),csr_matrix(X_test),y_train,y_test)
        if("log_reg_model" == model_id):
            return -log_reg_model(csc_matrix(X_train),csr_matrix(X_test),y_train,y_test)
        if("rand_forest_model" == model_id):
            return -rand_forest_model(csc_matrix(X_train),csr_matrix(X_test),y_train,y_test)      
        if("naive_bays_model" == model_id):
            return -naive_bays_model(csc_matrix(X_train),csr_matrix(X_test),y_train,y_test)
        
           
#load data from tf_matrix.csv
time_t = time.time()
f = open('tf_matrix.csv', 'r', encoding="utf-8")
data = pd.read_csv(f, header=0) 

#create pairs list
pairs  =  ["_I_E_","_N_S_", "_T_F_", "_J_P_"] 

#features are all word/tokens the occur in at least one text
features = list(data.keys())
features = features[:-5]

#X is all the occurances of the words for each text for each user
X = data[features]

#faster stucture..
#double check
#is normalization sufficient???
X = csr_matrix(X)
normalize(X, "l2")
X = csc_matrix(X)

classification_tests = tests()
Best_in_class = []
out_df = pd.DataFrame()
 

#how can you test 39 users when there is only 39 users for eahc personality
print ("Testing 39 users for each of the 16 meyers briggs personalities...\n")
c  =0
for item in pairs:  
    c = c+1
    y = data[item] 
    local_optimas_dec  = []
    local_optimas_reg  = []
    local_optimas_for  = []
    local_optimas_bays = []     
    i = 0
    #Range of the Gap note: optimiztion function fminbound finds the best value for the number of features
    gap_size = 8
    print(item, "Classification:")
    while(i<6):           
        
        #uses multithreading for a small to medium improvements in runtime
        
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
        
        # slower without the use of multithreading
        # local_optima_1 = fminbound(classification_tests.test_features,x1 = (i)*gap_size+1, x2 =(i+1)*gap_size+1 , args = ("dec_tree_model",X, y), full_output  = True, disp   =0)
        # local_optima_2 = fminbound(classification_tests.test_features,x1 = (i)*gap_size+1, x2 =(i+1)*gap_size+1 , args = ("log_reg_model",X, y), full_output  = True, disp   =0)
        # local_optima_3 = fminbound(classification_tests.test_features,x1 = (i)*gap_size+1, x2 =(i+1)*gap_size+1 , args = ("rand_forest_model",X, y), full_output  = True, disp   =0)
        # local_optima_4 = fminbound(classification_tests.test_features,x1 = (i)*gap_size+1, x2 =(i+1)*gap_size+1 , args = ("naive_bays_model",X, y), full_output  = True, disp   =0)
        
        #again the negation is wierd???
        #why is local_optima a list or tuple???
        local_optimas_dec.append((-local_optima_1[1], int(local_optima_1[0])))
        local_optimas_reg.append((-local_optima_2[1], int(local_optima_2[0])))
        local_optimas_for.append((-local_optima_3[1], int(local_optima_3[0])))
        local_optimas_bays.append((-local_optima_4[1], int(local_optima_4[0])))  


        #local_optima_1[1] is a score
        #and local_optima_1[0] is number of features used that gave the optimal score 
        print("Iteration:", i+1)
        print("Decision Tree Test:",  "Score:",-local_optima_1[1], "Features:", int(local_optima_1[0]))                       
        print("Logistic Regression Test:",  "Score:",-local_optima_2[1], "Features:", int(local_optima_2[0]))        
        print("Random Forest Test:", "Score:",-local_optima_3[1], "Features:", int(local_optima_3[0]))       
        print("Naive Bays Test:",  "Score:",-local_optima_4[1], "Features:", int(local_optima_4[0]),"\n")   
        
        #i controls the number of features
        #the optimizer finds the best number of features within bounds (i)*gap_size+1 and (i+1)*gap_size+1
        #then iterates again with i = i+1
        i += 1

    #find the best number features for each model
    local_optimas_dec.sort(reverse  = True) 
    local_optimas_reg.sort(reverse  = True)
    local_optimas_for.sort(reverse  = True)
    local_optimas_bays.sort(reverse  = True)
    
    out_df[item] = ["Decision Tree: "+"Features: "+str(local_optimas_dec[0][1])+" Accuracy: "+str(local_optimas_dec[0][0])
    ,"Logistic Regession: "+"Features: "+str(local_optimas_reg[0][1])+" Accuracy: "+str(local_optimas_reg[0][0])
    ,"Random Forest: "+ "Features: "+str(local_optimas_for[0][1])+" Accuracy: "+str(local_optimas_for[0][0])
    ,"Naive Bays: "+ "Features: "+str(local_optimas_bays[0][1])+" Accuracy: "+str(local_optimas_bays[0][0])]
    
    print(item, "Classification:")
    print("Best Predictor Function Scores:")
    print("Decision Tree:","Features:",local_optimas_dec[0][1],"Accuracy:",local_optimas_dec[0][0])
    print("Logistic Regession:","Features:",local_optimas_reg[0][1],"Accuracy:",local_optimas_reg[0][0])
    print("Random Forest:", "Features:",local_optimas_for[0][1],"Accuracy:",local_optimas_for[0][0])
    print("Naive Bays:", "Features:",local_optimas_bays[0][1],"Accuracy:",local_optimas_bays[0][0],"\n")
    
    Best_in_class.append([local_optimas_dec[0][0], 
                          local_optimas_reg[0][0], 
                          local_optimas_for[0][0], 
                          local_optimas_bays[0][0]])

results = []
classifiers = ["Decision Tree: ", "Logistic Regression: ", "Random Forest: ", "Naive Bays: "]
print("Complete Myers Briggs Prediction for individual Classifiers:")
for x in range (0,4):
    product =1
    for item in Best_in_class:
        product*= item[x]
    print(classifiers[x], product)
    results.append(str(classifiers[x])+ str(product))
print("\nMyers Briggs Prediction from the best of each Classifier:")
product  = 1
for item in Best_in_class:
    item.sort(reverse  = True)
    product*= item[0]
print(product)
out_df["Best in Class:"] = results
out_df["Overall Best"] = [str(product), "", "","" ]
out_file = open('results.csv', 'w', encoding="utf-8")
out_df.to_csv(out_file,index  = False)
out_file.close()    
print("Full compute time:",time.time()-time_t,"Seconds")








