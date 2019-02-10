import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from collections import defaultdict
import string
from nltk.stem.porter import PorterStemmer

def clean_review(review):
    """
    Removes punctuations, stopwords and returns an array of words
    """
    review = review.replace('&#34', '')
    p_stemmer = PorterStemmer()
    review = ''.join([c.lower() for c in review if c not in set(string.punctuation)])
    tokens = word_tokenize(review)
    tokens = [p_stemmer.stem(w) for w in tokens if w not in stopwords.words('english')]
    return ' '.join(tokens)

class Indexer:
    """
    Class to load data from file and obtain relevant data structures
    """
    def __init__(self):
        """
        Constructor
        """
        self.reviews = list()

    def read_file(self, filename):
        """
        Reads reviews from a specified file
        """
        f = open(filename)
        data = f.read()
        self.reviews = json.loads(data)
            

    def get_mappings(self,path_to_save_results):
        """
        Returns relevant data like vocab size, user list, etc after
        processing review data
        """
        
        ## retrieve users and items
        user_dict = dict()
        item_dict = dict()
        rating_list = []
        t_sum = defaultdict(int)
        t_count = defaultdict(int)
        
        for review in self.reviews:
            user = review['reviewerID']
            if user not in user_dict:
                nu = len(user_dict.keys())
                user_dict[user] = nu
                t_sum[user] += review['unixReviewTime']
                t_count[user] += 1
            
            item = review['asin']
            if item not in item_dict:
                nm = len(item_dict.keys())
                item_dict[item] = nm
        
        ## calculate mean review time for each user
        nu = len(user_dict.keys())
        t_mean = np.zeros(len(user_dict.keys()))
        user_list = [''] * nu
        for user in user_dict:
            idx = user_dict[user]
            user_list[idx] = user
            t_mean[idx] = t_sum[user]/t_count[user]

        ## build an item list            
        nm = len(item_dict.keys())
        item_list = [''] * nm
        for item in item_dict:
            idx = item_dict[item]
            item_list[idx] = item 

        ## create dictionary and clean reviews
        word_dictionary = dict()
        review_matrix = list()
        word_index = 0
        np.random.seed(5)
        review_map = list()
        item_reviews = [[] for o in range(len(item_dict))]
        
        indices = [i for i in range(len(self.reviews))]
        np.random.shuffle(indices)
        test_indices = {idx:1 for idx in indices[int(0.8*len(indices)):]}
        
        for index in range(len(self.reviews)):
            temp = clean_review(self.reviews[index]['reviewText'])
            review_map.append(
            {
                'user' : self.reviews[index]['reviewerID'],
                'movie' : self.reviews[index]['asin']
            })
            
            item_reviews[item_dict[self.reviews[index]['asin']]].append((temp, index))
            
            rating_list.append({'u': user_dict[self.reviews[index]['reviewerID']], 'm': item_dict[self.reviews[index]['asin']], 't': self.reviews[index]['unixReviewTime'], 'r':self.reviews[index]['overall']})
            arr = temp.split()
            review_matrix.append(arr)
            for ar in arr:
                ar = ar.strip()
                if ar not in word_dictionary:
                    word_dictionary[ar] = word_index
                    word_index += 1
        
        vocab_size = len(word_dictionary.keys())
        review_matrix = np.array(review_matrix)
         
        print('Number of reviews = ', len(self.reviews))
        np.save(path_to_save_results + 'word_dictionary.npy', [(k,word_dictionary[k]) for k in word_dictionary])
        return (vocab_size, user_list, item_list, review_matrix, review_map, user_dict, item_dict, rating_list, t_mean, item_reviews, word_dictionary,nu,nm,len(self.reviews), test_indices)