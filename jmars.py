import logging
from constants import *
from sampler import GibbsSampler
from optimize import optimizer
import numpy as np
import numpy.matlib
from indexer import Indexer

# Constants
MAX_ITER = 50
MAX_OPT_ITER = 1

def main():
    """
    Main function
    """
    # Download data for NLTK if not already done
    # nltk.download('all')

    # Read 
    np.random.seed(5)
    baseline = False  ## Make this true if you want to run the baseline, which is a simple latent factor model
    path_to_save_results = './test/'
    
    imdb = Indexer()
    imdb_file = 'data/clothing_data_small.json'  ## path to data file
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    logging.info('Reading file %s' % imdb_file)
    imdb.read_file(imdb_file)
    logging.info('File %s read' % imdb_file)
    
    (vocab_size, 
        user_list,  # remove
        movie_list, 
        review_matrix, 
        review_map, 
        user_dict, 
        movie_dict, 
        rating_list, 
        t_mean, 
        movie_reviews, 
        word_dictionary,
        U, M, R, test_indices) = imdb.get_mappings(path_to_save_results)
    

    mul_factor = 0.1
    ## Initialize
    alpha_vu = np.random.normal(0,sigma_u,(U, K)) * mul_factor
    alpha_bu = np.random.normal(0,sigma_u,(U, 1)) * mul_factor
    alpha_tu = np.random.normal(0,sigma_u,(U, A)) * mul_factor
    
    
    # User
    v_u = np.random.normal(0,sigma_u,(U, K)) * mul_factor      # Latent factor vector
    b_u = np.random.normal(0,sigma_bu,(U, 1)) * mul_factor      # Common bias vector
    theta_u = np.random.normal(0,sigma_ua,(U, A)) * mul_factor  # Aspect specific vector
    
    # Movie
    v_m = np.random.normal(0,sigma_m,(M, K)) * mul_factor      # Latent factor vector
    b_m = np.random.normal(0,sigma_bm,(M, 1)) * mul_factor      # Common bias vector
    theta_m = np.random.normal(0,sigma_ma,(M, A)) * mul_factor  # Aspect specific vector
    
    # Common bias
    b_o = np.random.normal(0,sigma_b0)  * mul_factor
    
    # Scaling Matrix
    M_a = np.random.normal(0,sigma_Ma,(A, K))  * mul_factor
    
    params = numpy.concatenate((alpha_vu.flatten('F'), 
                                    v_u.flatten('F'), 
                                    alpha_bu.flatten('F'), 
                                    b_u.flatten('F'), 
                                    alpha_tu.flatten('F'), 
                                    theta_u.flatten('F'), 
                                    v_m.flatten('F'), 
                                    b_m.flatten('F'), 
                                    theta_m.flatten('F'), 
                                    M_a.flatten('F'), 
                                    np.array([b_o]).flatten('F')))

    save_test_rmse = []
    # Get number of users and movies
    Users = len(user_list)
    Movies = len(movie_list)
    logging.info('No. of users U = %d' % Users)
    logging.info('No. of movies M = %d' % Movies)


    # change gibbs sampler initialization
    gibbs_sampler = GibbsSampler(vocab_size,
                                    review_matrix,
                                    rating_list,
                                    movie_dict,
                                    user_dict,
                                    movie_reviews,
                                    word_dictionary
                                    ,U, M, R, test_indices)


    # Run Gibbs EM
    for it in range(1,MAX_ITER+1):
        print('Running iteration %d of Gibbs EM' % it)
        print('Running E-Step - Gibbs Sampling')

        if baseline != True:
            Nums,Numas,Numa = gibbs_sampler.run(vocab_size, 
                                                review_matrix, 
                                                rating_list, 
                                                user_dict, 
                                                movie_dict, 
                                                movie_reviews, 
                                                word_dictionary, 
                                                t_mean, 
                                                params, test_indices, path_to_save_results)
        else:
            Nums = np.zeros((R,2))
            Numas = np.zeros((R,A,2))
            Numa = np.zeros((R,A))
        print('Running M-Step - Gradient Descent')
        for i in range(1,MAX_OPT_ITER+1):
            params,save_test_rmse = optimizer(Nums,Numas,Numa,rating_list,t_mean,params,U,M,R,test_indices,save_test_rmse)
            np.save(path_to_save_results+'params.npy',params)
            np.save(path_to_save_results+'performance_notime_medium_noreg_seed5.npy',save_test_rmse)
    
if __name__ == "__main__":
    main()