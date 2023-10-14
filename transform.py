
import pandas as pd
import nltk
import csv
import itertools
import time
import random

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from ordered_set import OrderedSet


#downloads needed to work locally
nltk.download('punkt')
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')

#the number of users of the least occuring personality type
NUM_USERS = 39

#used to keep consistent choices of random users each runtime which keeps all outputs consistent
RSEED  = 5

def load_data():
    """
    load and return the data_by_type and data_by_type_full variables populated from mbti_1.csv
    data_by_type_full is a dictionary of personality string keys like "ENTJ" to a list of the corresposnding users of that personality type
    each user is a list of 50 posts
    data_by_type is a dictionary of personality string keys like "ENTJ" to list of 39 random users of that type selected randomly from data_by_type_full
    each user is a list of 50 posts
    why 39 users??? that is the number of users of the least occuring personailty type (NUM_USERS = 39)
    """

    f = open('mbti_1.csv', newline='', encoding="utf-8")
    data = csv.reader(f)
    # next(data) cuts the first row which are column labels
    next(data)
    data_by_type_full = dict()
    for row in data:   
        if ((row[0] not in data_by_type_full.keys())):
            data_by_type_full[row[0]] = []            
        data_by_type_full[row[0]].append(row[1].split("|||"))
    data_by_type = dict()
    for key in data_by_type_full.keys():
        data_by_type[key] = random.sample(data_by_type_full[key], NUM_USERS)
    f.close()   
    return data_by_type, data_by_type_full



def tf_full(pairs, data_by_type):
    """create a list of term frequencies and there coresponding personalities"""
    #the OrderedSet (word_bank) is used because all the unqiue terms need to be in a consistent order
    word_bank = OrderedSet()
    #list of word occurance counts corresponding to the words_bank for every user
    word_occurances = []
    #this becomes a list of 4 lists of personality bits for each personality pair
    pair_types = [[],[],[],[]]
    #this is a list of integers corresponding to personality type int (0-15)
    types = []
    #bag of words for every user in every personailty
    bag_of_words_dict = dict()
    #populate the bag of words and word_bank
    for w,x,y,z in itertools.product(pairs[0],pairs[1],pairs[2],pairs[3]): 
        bag_of_words_dict[w+x+y+z] = []
        for i in range (0,NUM_USERS): 
            bag_of_words_dict[w+x+y+z].append([])
            update_word_bank(bag_of_words_dict, word_bank ,data_by_type,i, w+x+y+z)           
    c = 0    
    #populate rows of corresponding (word_occurances, pairs, and types)
    for w,x,y,z in itertools.product(pairs[0],pairs[1],pairs[2],pairs[3]): 
        for i in range (0,NUM_USERS):   
            #init word occurances for a user
            user_word_occurances = [0] * len(word_bank)
            #populate user_word_occurances
            update_word_occurances(bag_of_words_dict, word_bank, user_word_occurances, i, w+x+y+z) 
            #append user_word_occurances for a user to word_occurances
            word_occurances.append(user_word_occurances)
            #this populates pair_types
            populate_pairs(pair_types, pairs, w+x+y+z)
            # c is an integer to represent personaility with an integer (0-15)
            types.append(c)    
        c = c+1               
    to_csv("tf_matrix.csv", word_occurances, word_bank, pair_types, types)
    




def update_word_bank(bag_of_words_dict, word_bank,data_by_type,x, key):
    """look through a single users posts and update the word bank and the bag_of_words_dict with the tokens"""
    bag_of_words_dict[key][x] = []
    for post in data_by_type[key][x]:
        tokens_1 = word_tokenize(post)
        #ignore cases and remove stopwords
        tokens_2 = [WordNetLemmatizer().lemmatize(token.lower()) for token in tokens_1 if token.isalpha() 
                    and WordNetLemmatizer().lemmatize(token.lower()) not in stopwords.words('english')]  
        word_bank.update(tokens_2)
        bag_of_words_dict[key][x].extend(tokens_2)




def update_word_occurances(bag_of_words_dict, word_bank, user_word_occurances,x, key):
    """
    populate user_word_occurances, a list of counts where each count represents the number of term ocurrances
    for the terms in the order of the word_bank
    """
    i = 0
    for word in word_bank:
        for instance in bag_of_words_dict[key][x]:
            if instance == word:
                user_word_occurances[i] = user_word_occurances[i]+ 1 
        i += 1 


def populate_pairs(pair_types, pairs, key):
    """this populates the 4 pair_types of the user each represented by a 1 or 0""" 
    i =0
    for item in pairs:
        if(item[0] == key[i]): 
            pair_types[i].append(1)
        else:
            pair_types[i].append(0)
        i = i+1


def to_csv(file_name, word_occurances, word_bank, pair_types,  types): 
    """output term frequencies and personality columns to tf_matrix.csv"""
    #the intial columns are the word occorances
    df = pd.DataFrame(data=word_occurances, columns=list(word_bank))  
    #these columns are the personailty pair columns 
    df["_I_E_"] = pair_types[0]
    df["_N_S_"] = pair_types[1]
    df["_T_F_"] = pair_types[2]
    df["_J_P_"] = pair_types[3]
    #this column is the full type column ranging from int (0-15)
    df["_Type_"] = types
    f = open(file_name, 'w', encoding='utf-8')
    df.to_csv(f,index  = False)
    f.close()    


def main():
    time_t = time.time()
    random.seed(RSEED)

    data_by_type, data_by_type_full = load_data()                                
    pairs  = [['I', 'E'],['N', 'S'],['T', 'F'],['J', 'P']]

    print("Orginal counts per Personality:")
    for w,x,y,z in itertools.product(pairs[0],pairs[1],pairs[2],pairs[3]):
        print(w+x+y+z, len(data_by_type_full[w+x+y+z]))
        
    tf_full(pairs, data_by_type)

    print("Done")
    print("Full compute time:",float((time.time() - time_t)/60), "Minutes")



if __name__ == "__main__":
    main()



#both: bag of words split...

#version with:
# tokens_2 = [WordNetLemmatizer().lemmatize(token.lower()) for token in tokens_1 
#             if WordNetLemmatizer().lemmatize(token.lower()) in word_bank]
#Full compute time: 6.870408161481222 Minutes


#version with:
# tokens_2 = [WordNetLemmatizer().lemmatize(token.lower()) for token in tokens_1 
#             if token.isalpha() and WordNetLemmatizer().lemmatize(token.lower()) not in stopwords.words('english')] 
# Full compute time: 9.52508595387141 Minutes


#version with updates bag of words:
#Full compute time: 6.733085672060649 Minutes


#misc:
#why is a variables like bag of words not updated in the above function to save processing??? 

#the bag of words and the word_bank could be sorted
#the wordbank only needs to be sorted once but the bag of words is per user 

#why is the conditon in the following function search the word bank???
#Is it faster than confirming if its not in stopwords and is a alpha numerical???

#can introduce a new variable, bag_of_words_dict that adds all the copies of all the words/terms tokens_2
#it is a dictionary organized like data_by_type