
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

#downloads needed to work locally
nltk.download('punkt')
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')


#random seed
rseed = 10
random.seed(rseed)

# load and return the data_by_type variable populated from the mbti_1.csv
# data_by_type is a dictionary of personality types to list of users
# each user is defined by a list of 50 posts that they made
def load_data():
    f = open('mbti_1.csv', newline='', encoding="utf-8")
    data = csv.reader(f)
    # what does next(data) do???
    # this cuts the first row which are column labels
    next(data)
    data_by_type_full = dict()
    #populate data by type
    for row in data:   
        if ((row[0] not in data_by_type_full.keys())):
            data_by_type_full[row[0]] = []            
        posts = row[1].split("|||")
        user  = []
        #this loop and directly below it is not needed...
        #just use data_by_type[row[0]].append(row[1].split("|||"))
        for post in posts:
            user.append(post)
        data_by_type_full[row[0]].append(user) 

    #this is data_by_type_full but only 39 randomly selected users for each personality
    #there are exactly 39 users for each personailty type because that is the minimum number of users...
    #for the 16 persoanlity types in the original data
    data_by_type = dict()
    for key in data_by_type_full.keys():
        data_by_type[key] = random.sample(data_by_type_full[key], 39)
    f.close()   
    return data_by_type, data_by_type_full




def tf_full(pairs, data_by_type):
    #set of all words amoung every users post   
    word_bank = set()
    #count of words occurances for every user
    word_occurances = []
    #this becomes a list of 4 lists of personality bits 
    #this could instead be a list of lists of 4 personality bits
    pair_types = [[],[],[],[]]
    types = []
    for w,x,y,z in itertools.product(pairs[0],pairs[1],pairs[2],pairs[3]):  
        #there are exactly 39 users for each personailty type because that is the minimum number of users...
        #for the 16 persoanlity types in the original data
        #popularity bias should not be introduced
        for i in range (0,39): 
            update_word_bank(word_bank ,data_by_type,i, w+x+y+z)           
    c = 0    

    #for each personality type
    for w,x,y,z in itertools.product(pairs[0],pairs[1],pairs[2],pairs[3]): 
        #there are exactly 39 users for each personailty type
        #note: this is the minimum count for a personality type in the original dataset
        for i in range (0,39):   
            #create a dictionary with the keys the same as the word_bank set
            #and the values (the # of accorances initially set to 0)
            group_word_occurances = dict.fromkeys(list(word_bank),0)  

            #populate the real count values in group_word_occurances...
            update_word_occurances(word_bank, group_word_occurances, i, data_by_type, w+x+y+z) 

            #extract only the occurances without the keys and append to word_occurances
            #Do the values hold consistent order for each iteration??? (yes)
            #https://stackoverflow.com/questions/835092/python-dictionary-are-keys-and-values-always-the-same-order#:~:text=Yes%2C%20what%20you%20observed%20is,order%20as%20the%20corresponding%20lists.
            word_occurances.append(list(group_word_occurances.values()))

            #pairs is a list of the personaility duos
            #w+x+y+z is a single string representing a single personality
            #pair_types is a list of booleans to represent each personality pair

            #this populates pair_types
            #add the same value to pair_types the for all 39 sample users of type w+x+y+z
            populate_pairs(pair_types, pairs, w+x+y+z)

            # c is an integer to represent personaility (0-15)
            # same for all 39 sample users of type w+x+y+z
            types.append(c)    
        # starts at 0 and goes to 15   
        c = c+1        
    #to_csv          
    to_csv("tf_matrix.csv", word_occurances, word_bank, pair_types, types)
    
    
def update_word_bank(word_bank,data_by_type,x, key):
    #look through a single users posts and update the word bank

    #here it is possible to randomly select the users without replacement rather than taking the first 39 users
    #or the order can simply be scrambled before hand...
    for post in data_by_type[key][x]:
        tokens_1 = word_tokenize(post)
        tokens_2 = [WordNetLemmatizer().lemmatize(token.lower()) for token in tokens_1 if token.isalpha() 
                    and WordNetLemmatizer().lemmatize(token.lower()) not in stopwords.words('english')]
        word_bank.update(tokens_2)
        
                
def update_word_occurances(word_bank, group_word_occurances,x, data_by_type,key):       
    bag_of_words = [] 
    #populate bag of words
    for post in data_by_type[key][x]:
        #seperate all the words and store in tokens
        tokens_1 = word_tokenize(post)
        #check if the token is in word bank set for each token
        #this does not make sense because the tokens have not been lemmatized, isalpha ect. like above
        tokens_2 = [WordNetLemmatizer().lemmatize(token.lower()) for token in tokens_1 
                    if WordNetLemmatizer().lemmatize(token.lower()) in word_bank]
        bag_of_words.extend(tokens_2)
        
    # go through the word bank set and find each occurance in the bag of words 
    # and update the value for group_word_occurances
    # note: order of the word bank does not matter since the group_word_occurances is a dictionary  

    # this might be faster if the bag of words were sorted then counted and put into
    # group_word_occurances O(n^2) vs O(nlogn) + O(n)

    for word in word_bank:
        for instance in bag_of_words:
            if instance == word:
                group_word_occurances[word] = group_word_occurances[word]+ 1  

#note: pair_types is a list of booleans that determine the personality combination
def populate_pairs(pair_types, pairs, key): 
    c =0
    for item in pairs:
        if(item[0] in key): 
            pair_types[c].append(1)
        else:
            pair_types[c].append(0)
        c = c+1

           
def to_csv(file_name, word_occurances, word_bank, pair_types,  types):   
    #note: word_occurances does not necessarily line up with list(word_bank)
    df = pd.DataFrame(data=word_occurances, columns=list(word_bank))  
    #these columns are the personailty columns 
    df["_I_E_"] = pair_types[0]
    df["_N_S_"] = pair_types[1]
    df["_T_F_"] = pair_types[2]
    df["_J_P_"] = pair_types[3]
    #this column is the full type column ranging (0-15)
    df["_Type_"] = types
    f = open(file_name, 'w', encoding='utf-8')
    df.to_csv(f,index  = False)
    f.close()    
    
time_t = time.time()
#note: 
data_by_type, data_by_type_full = load_data()                                
pairs  = [['I', 'E'],['N', 'S'],['T', 'F'],['J', 'P']]


print("Data per Personality:")
for w,x,y,z in itertools.product(pairs[0],pairs[1],pairs[2],pairs[3]):
    print(w+x+y+z, len(data_by_type_full[w+x+y+z]))
    


tf_full(pairs, data_by_type)
print("Full compute time:",time.time() - time_t,"Seconds")


#idea: the 39 users of each personality should be chosen randomly...