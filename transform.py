
import pandas as pd
import nltk
import csv
import itertools
import time
from nltk.corpus import words
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('punkt')
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data():
    #load and return the data_by_type variable
    #this is a dictionary of personality types to list of users
    #each user is a list of posts
    f = open('mbti_1.csv', newline='', encoding="utf-8")
    data = csv.reader(f)
    #what does next(data) do???
    next(data)
    data_by_type = {}
    for row in data:   
        if ((row[0] not in data_by_type.keys())):
            data_by_type[row[0]] = []            
        posts = row[1].split("|||")
        user  = []
        for post in posts:
            user.append(post)
        data_by_type[row[0]].append(user)       
    f.close()   
    return data_by_type

# create a term frequency table with the terms as columns and each other row in the table is
# a list of frequencies for the full word bank
# this can probably be optmized
def tf_full(pairs, data_by_type):   
    word_bank = set()
    #user X the word occurances for that user
    word_occurances = []
    pair_types = [[],[],[],[]]
    types = []
    for w,x,y,z in itertools.product(pairs[0],pairs[1],pairs[2],pairs[3]):  
        #there are exactly 39 users for each personailty type
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
            word_occurances.append(list(group_word_occurances.values()))

            #pairs is a list of the personaility duos
            #w+x+y+z is a single string representing a single personality
            #pair_types is a list of booleans to represent each personality pair
            #this populates pair_types

            #add the same value to pair_types the for all 39 sample users of type w+x+y+z
            populate_pairs(pair_types, pairs, w+x+y+z)

            # c is an integer to represent personaility
            # same for all 39 sample users of type w+x+y+z
            types.append(c)    
        # starts at 0 and goes to 15   
        c = c+1        
    #to_csv          
    to_csv("tf_matrix.csv", word_occurances, word_bank, pair_types, types)
    
    
def update_word_bank(word_bank,data_by_type,x, key):
    #look through the users posts and update the word bank
    #make sure words are not stopwords and must be word .isalpha()
    #make sure the lematized version is used
    #may want to roll back this update to possiblly improve perf...
    #there may be an ordering problem with dictionary keys...
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
    # this is time consuming an inefficent  
    # this does not make sense because insatnces in bag of words has not been lemmatized, isalpha ect. like above               
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
    #the list of intial columns is the word bank  
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
data_by_type = load_data()                                
pairs  = [['I', 'E'],['N', 'S'],['T', 'F'],['J', 'P']]


print("Data per Personality:")
for w,x,y,z in itertools.product(pairs[0],pairs[1],pairs[2],pairs[3]):
    print(w+x+y+z, len(data_by_type[w+x+y+z]))
    


tf_full(pairs, data_by_type)
print("Full compute time:",time.time() - time_t,"Seconds")


#Big Problem:
#Capital versions and lower case versions of words are both in the final tf_matrix...