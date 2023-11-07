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


# Downloads needed to work locally:
nltk.download('punkt')
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')

# The number of users for the least occuring personality type:
NUM_USERS = 39
# The number to seed in order to keep consistent runtimes:
SEED_INT = 5



def load_data():
    """
    Load and return the data_by_type and data_by_type_full variables populated from "mbti_1.csv".
    "data_by_type_full" is a dictionary of personality string keys like "ENTJ" to a list of the corresposnding users of that personality type.
    Each user is a list of 50 posts.
    "data_by_type" is a dictionary of personality string keys like "ENTJ" to list of 39 random users of that type selected randomly from data_by_type_full
    Each user is a list of 50 posts.
    Why 39 users??? That is the number of users of the least occuring personailty type (NUM_USERS = 39)
    See README.md for reason to truncate the number of users in each category to 39.
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
    """Create a list of users term frequencies and there coresponding personalities."""
    # This is used to set the consistent order of terms for a users word/term count vector (see "user_word_occurances" below).
    word_bank = OrderedSet()
    # For every user, a word count vector with the term order corresponding to the order of words in "word_bank".
    word_occurances = []
    # This becomes a list of 4 lists of personality bits for each personality pair.
    pair_types = [[],[],[],[]]
    # This is a list of integers (0-15) corresponding to personality type. 
    # It is the same inforamtion as pair_types but in a different format.
    types = []
    # Every personality key (like "ENTJ") to a list, which is a bag_of_words for each user of that personality type.
    bag_of_words_dict = dict()
    # Populate the "bag_of_words_dict" and "word_bank".
    for w,x,y,z in itertools.product(pairs[0],pairs[1],pairs[2],pairs[3]): 
        bag_of_words_dict[w+x+y+z] = []
        for i in range (0,NUM_USERS): 
            bag_of_words_dict[w+x+y+z].append([])
            update_word_bank(bag_of_words_dict, word_bank ,data_by_type,i, w+x+y+z) 

    # c is an integer to represent personaility types (0-15)          
    c = 0    
    # Populate rows of word_occurances, pair_types, and types
    for w,x,y,z in itertools.product(pairs[0],pairs[1],pairs[2],pairs[3]): 
        for i in range (0,NUM_USERS):   
            user_word_occurances = [0] * len(word_bank)
            update_word_occurances(bag_of_words_dict, word_bank, user_word_occurances, i, w+x+y+z) 
            word_occurances.append(user_word_occurances)
            populate_pairs(pair_types, pairs, w+x+y+z)
            types.append(c)    
        c = c+1               
    to_csv("tf_matrix.csv", word_occurances, word_bank, pair_types, types)
    




def update_word_bank(bag_of_words_dict, word_bank,data_by_type,x, key):
    """Look through a single users posts and update the "word_bank" and the "bag_of_words_dict" with the tokens."""
    bag_of_words_dict[key][x] = []
    for post in data_by_type[key][x]:
        tokens_1 = word_tokenize(post)
        # Ignore cases and remove stopwords
        tokens_2 = [WordNetLemmatizer().lemmatize(token.lower()) for token in tokens_1 if token.isalpha() 
                    and WordNetLemmatizer().lemmatize(token.lower()) not in stopwords.words('english')]  
        word_bank.update(tokens_2)
        bag_of_words_dict[key][x].extend(tokens_2)




def update_word_occurances(bag_of_words_dict, word_bank, user_word_occurances,x, key):
    """
    Populate "user_word_occurances", a users count vector with the term order corresponding to the order of words in "word_bank".
    """
    i = 0
    for word in word_bank:
        for instance in bag_of_words_dict[key][x]:
            if instance == word:
                user_word_occurances[i] = user_word_occurances[i]+ 1 
        i += 1 


def populate_pairs(pair_types, pairs, key):
    """This populates the 4 pair_types of the user (each pair has the value 1 or 0)""" 
    i =0
    for item in pairs:
        if(item[0] == key[i]): 
            pair_types[i].append(1)
        else:
            pair_types[i].append(0)
        i = i+1


def to_csv(file_name, word_occurances, word_bank, pair_types,  types): 
    """Output term frequencies and personality columns to tf_matrix.csv"""
    # The intial columns are the word occorances.
    df = pd.DataFrame(data=word_occurances, columns=list(word_bank))  
    # These columns are the personailty pair columns.
    df["_I_E_"] = pair_types[0]
    df["_N_S_"] = pair_types[1]
    df["_T_F_"] = pair_types[2]
    df["_J_P_"] = pair_types[3]
    # This column is the full type column ranging from int (0-15)
    df["_Type_"] = types
    f = open(file_name, 'w', encoding='utf-8')
    df.to_csv(f,index  = False)
    f.close()    


def main():
    time_t = time.time()

    # Seed for consistent results across runtimes:
    random.seed(SEED_INT)

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

