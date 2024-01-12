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

from sklearn.feature_extraction.text import TfidfVectorizer

# Downloads needed to work locally:
nltk.download('punkt')
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')

# The number of users for the least occuring personality type:
NUM_USERS = 39
# Global random seed: 
SEED_INT = 5


def load_data():
    """
    Load and return the data_by_type and data_by_type_full variables populated from "mbti_1.csv".
    "data_by_type_full" is a dictionary of personality string keys like "ENTJ" to a list of the corresponding users of that personality type.
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
        data_by_type_full[row[0]].append(row[1].replace("|||", " "))
    data_by_type = dict()
    for key in data_by_type_full.keys():
        data_by_type[key] = random.sample(data_by_type_full[key], NUM_USERS)
    f.close()   
    return data_by_type, data_by_type_full


# LOOK: data_by_type is a dictionary of personality type to list of users
# Each user is a combined string of their 50 posts 

# TfidfTransformer.fit_transform()
# Takes a list of strings where each string is all the text written by a single user
# returns the Tf-idf-weighted document-term matrix

# How do I convert this into a dataframe where each column is a term???
# vectorizer.get_feature_names_out() gives the terms in the correct order.


def tf_full(pairs, data_by_type):
    """Create a list of users term frequencies and there corresponding personalities."""
    corpus_list = []
    pair_types = [[],[],[],[]]

    for key in data_by_type.keys():
        for _ in data_by_type[key]:
            for pair, char, index in zip(pairs, key, range(4)):
                if char == pair[0]:
                    pair_types[index].append(1)
                else:
                    pair_types[index].append(0)
        

        corpus_list.extend(data_by_type[key])

    vectorizer = TfidfVectorizer(lowercase = True, stop_words = "english")
    document_term_matrix = vectorizer.fit_transform(corpus_list)
    terms = vectorizer.get_feature_names_out()

    #For each user, in the order of "terms" there needs to be a list of term frequencies (sparse)
    to_csv("tf_matrix.csv", document_term_matrix.toarray(), terms, pair_types)


def to_csv(file_name, word_occurances, terms, pair_types): 
    """Output term frequencies and personality columns to tf_matrix.csv"""
    # The intial columns are the word occurances.
    df = pd.DataFrame(data=word_occurances, columns= terms)  
    # These columns are the personailty pair columns.
    df["_I_E_"] = pair_types[0]
    df["_N_S_"] = pair_types[1]
    df["_T_F_"] = pair_types[2]
    df["_J_P_"] = pair_types[3]

    f = open(file_name, 'w', encoding='utf-8')
    df.to_csv(f,index  = False)
    f.close()    


#LOOK: need to try using just basic document frequency instead of tf-idf in both projects
#LOOk: potential issue with this code is the wrong kind of lematization is used

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

