
import pandas as pd
import nltk
import csv
import itertools
import time
import random

#not used
from nltk.corpus import words

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from ordered_set import OrderedSet

#downloads needed to work locally
nltk.download('punkt')
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')



# load and return the data_by_type and data_by_type_full variables populated from mbti_1.csv
# They are both a dictionary of personality types to list of users of that personality
# each user is defined by a list of 50 posts that they made
def load_data():
    f = open('mbti_1.csv', newline='', encoding="utf-8")
    data = csv.reader(f)
    # next(data) cuts the first row which are column labels
    next(data)
    data_by_type_full = dict()
    for row in data:   
        if ((row[0] not in data_by_type_full.keys())):
            data_by_type_full[row[0]] = []            
        data_by_type_full[row[0]].append(row[1].split("|||"))
    #Only a randomly selelcted 39 users of each for the 16 personality type are used to train the models
    #That is the number of users for the rarest personality
    #This way there is no popularity bias built into the model 
    data_by_type = dict()
    for key in data_by_type_full.keys():
        data_by_type[key] = random.sample(data_by_type_full[key], 39)
    f.close()   
    return data_by_type, data_by_type_full



#Ouputs the data to tf_matrix in a format ready for the analysis program
def tf_full(pairs, data_by_type):
    #the word_bank OrderedSet is used because all the unqiue terms need to be in a consistent order
    word_bank = OrderedSet()
    #count of words occurances for every user
    #the counts themselves are in the order of the word bank
    word_occurances = []
    #this becomes a list of 4 lists of personality bits for each personality pair
    pair_types = [[],[],[],[]]
    #this is a list of integers corresponding to personality type int (0-15)
    types = []
    #populate word_bank
    for w,x,y,z in itertools.product(pairs[0],pairs[1],pairs[2],pairs[3]):  
        for i in range (0,39): 
            update_word_bank(word_bank ,data_by_type,i, w+x+y+z)           
    c = 0    
    #populate word_occurances and pairs
    for w,x,y,z in itertools.product(pairs[0],pairs[1],pairs[2],pairs[3]): 
        for i in range (0,39):   
            #init word occurances for a user
            group_word_occurances = [0] * len(word_bank)
            #populate group_word_occurances
            update_word_occurances(word_bank, group_word_occurances, i, data_by_type, w+x+y+z) 
            #append group_word_occurances for a user to word_occurances
            word_occurances.append(group_word_occurances)
            #pairs is a list of the personality pairs with 16 combinations
            #w+x+y+z is a single string representing a single personality
            #this populates pair_types
            populate_pairs(pair_types, pairs, w+x+y+z)
            # c is an integer to represent personaility with an integer (0-15)
            types.append(c)    
        # starts at 0 and goes to 15   
        c = c+1        
    #this is the output of the data prep progam used for the start of the analysis program       
    to_csv("tf_matrix.csv", word_occurances, word_bank, pair_types, types)
    

#look through a single users posts and update the word bank 
def update_word_bank(word_bank,data_by_type,x, key):
    for post in data_by_type[key][x]:
        tokens_1 = word_tokenize(post)
        tokens_2 = [WordNetLemmatizer().lemmatize(token.lower()) for token in tokens_1 if token.isalpha() 
                    and WordNetLemmatizer().lemmatize(token.lower()) not in stopwords.words('english')]  
        word_bank.update(tokens_2)
        
#return a list of word counts in the order of the word_bank       
def update_word_occurances(word_bank, group_word_occurances,x, data_by_type,key): 
    #all the words that a user used, copies allowed
    bag_of_words = [] 
    #populate bag of words
    for post in data_by_type[key][x]:
        tokens_1 = word_tokenize(post)
        tokens_2 = [WordNetLemmatizer().lemmatize(token.lower()) for token in tokens_1 
                    if WordNetLemmatizer().lemmatize(token.lower()) in word_bank]
        bag_of_words.extend(tokens_2)

    #populate group_word_occurances in the order of the terms in word_bank set
    i = 0
    for word in word_bank:
        for instance in bag_of_words:
            if instance == word:
                group_word_occurances[i] = group_word_occurances[i]+ 1 
        i += 1 

#note: pair_type is a list of 4 lists of personality bits...
#for the users personailty selection for the given pair
def populate_pairs(pair_types, pairs, key): 
    c =0
    for item in pairs:
        if(item[0] == key[c]): 
            pair_types[c].append(1)
        else:
            pair_types[c].append(0)
        c = c+1

#output to tf_matrix.csv for the analysis program
def to_csv(file_name, word_occurances, word_bank, pair_types,  types):   
    df = pd.DataFrame(data=word_occurances, columns=list(word_bank))  
    #these columns are the personailty columns 
    df["_I_E_"] = pair_types[0]
    df["_N_S_"] = pair_types[1]
    df["_T_F_"] = pair_types[2]
    df["_J_P_"] = pair_types[3]
    #this column is the full type column ranging from int(0-15)
    df["_Type_"] = types
    f = open(file_name, 'w', encoding='utf-8')
    df.to_csv(f,index  = False)
    f.close()    
    

#main program...
time_t = time.time()

#consistent seed
rseed = 5
random.seed(rseed)

data_by_type, data_by_type_full = load_data()                                
pairs  = [['I', 'E'],['N', 'S'],['T', 'F'],['J', 'P']]

print("Orginal data per Personality:")
for w,x,y,z in itertools.product(pairs[0],pairs[1],pairs[2],pairs[3]):
    print(w+x+y+z, len(data_by_type_full[w+x+y+z]))
    
tf_full(pairs, data_by_type)
print("Done")
print("Full compute time:",time.time() - time_t,"Seconds")
