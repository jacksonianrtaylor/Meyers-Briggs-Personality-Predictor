#Jackson Taylor
#Student ID: 00001590147
#CSCI 182: Web and Data Mining (35622)
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


def load_data():
    f = open('mbti_1.csv', newline='', encoding="utf-8")
    data = csv.reader(f)
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

        
def tf_full(pairs, data_by_type):   
    word_bank = set()
    word_occurances = []
    pair_types = [[],[],[],[]]
    types = []
    for w,x,y,z in itertools.product(pairs[0],pairs[1],pairs[2],pairs[3]):    
        for i in range (0,39): 
            update_word_bank(word_bank ,data_by_type,i, w+x+y+z)           
    c = 0    
    for w,x,y,z in itertools.product(pairs[0],pairs[1],pairs[2],pairs[3]): 
        for i in range (0,39):   
            group_word_occurances = dict.fromkeys(list(word_bank),0)  
            update_word_occurances(word_bank, group_word_occurances, i, data_by_type, w+x+y+z) 
            word_occurances.append(list(group_word_occurances.values()))
            populate_pairs(pair_types, pairs, w+x+y+z)
            types.append(c)       
        c = c+1                    
    to_csv("tf_matrix.csv", word_occurances, word_bank,pair_types, types)
    
    
def update_word_bank(word_bank,data_by_type,x, key):
    for post in data_by_type[key][x]:
        tokens = word_tokenize(post)
        tokens = [token for token in tokens if token.isalpha() and token not in stopwords.words('english')]
        tokens  = [WordNetLemmatizer().lemmatize(token) for token in tokens]
        word_bank.update(tokens)
        
                
def update_word_occurances(word_bank, group_word_occurances,x, data_by_type,key):       
    bag_of_words = []  
    for post in data_by_type[key][x]:
        tokens = word_tokenize(post)
        tokens = [token for token in tokens if token in word_bank]
        bag_of_words.extend(tokens)
                            
    for word in word_bank:
        for instance in bag_of_words:
            if instance == word:
                group_word_occurances[word] = group_word_occurances[word]+ 1  
                
def populate_pairs(pair_types, pairs, key): 
    c =0
    for item in pairs:
        if(item[0] in key): 
            pair_types[c].append(1)
        else:
            pair_types[c].append(0)
        c = c+1

           
def to_csv(file_name, word_occurances, word_bank, pair_types,  types):       
    df = pd.DataFrame(data=word_occurances, columns=list(word_bank))     
    df["_I_E_"] = pair_types[0]
    df["_N_S_"] = pair_types[1]
    df["_T_F_"] = pair_types[2]
    df["_J_P_"] = pair_types[3]
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
