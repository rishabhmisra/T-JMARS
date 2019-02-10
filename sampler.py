from constants import *
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz

def dev_t(t, tu_mean):
    return np.sign(t-tu_mean)*abs(t-tu_mean)**beta
#    return 0.0

# Joint aspect distribution
def joint_aspect(u, m, t, t_mean_u):
    """
    Returns the joint aspect distribution
    """
    
    num_theta_uma = np.exp(theta_u[u] + dev_t(t, t_mean_u)*alpha_tu[u] + theta_m[m])
    theta_uma = np.divide(num_theta_uma.T,num_theta_uma.sum()).T

    return theta_uma

def predicted_rating(u, m, t, t_mean_u):
    """
    Computes the predicted rating for user u on movie m
    """

    theta_uma = joint_aspect(u, m, t, t_mean_u)
    M_sum = np.dot(theta_uma, M_a)

    v_ut = v_u[u] + dev_t(t, t_mean_u)*alpha_vu[u]
    b_ut = b_u[u] + dev_t(t, t_mean_u)*alpha_bu[u]
    r = np.dot(np.dot(v_ut, np.diag(M_sum)), v_m[m].T) + b_o + b_ut + b_m[m]  
    return r

def predicted_aspect_rating(u, m, a, t, t_mean_u):
    """
    Computes the predicted rating for user u on movie m and aspect a
    """
    v_ut = v_u[u] + dev_t(t, t_mean_u)*alpha_vu[u]
    b_ut = b_u[u] + dev_t(t, t_mean_u)*alpha_bu[u]
    
    r = np.dot(np.dot(v_ut, np.diag(M_a[a])), v_m[m].T) + b_o + b_ut + b_m[m]
    return r

def aspect_sentiment_probability(s, u, m, a, t, t_mean_u):
    """
    Computes the probability for a sentiment s on aspect a 
    for user u on movie m
    """
    ruma = predicted_aspect_rating(u,m,a,t, t_mean_u)
    prob_suma = 1.0 / (1.0 + np.exp(-s*(c*ruma - b)))
    return prob_suma

def aggregate_sentiment_probability(s, u, m, t, t_mean_u):
    """
    Computes the probability for aggregate sentiment s 
    for user u and movie m
    """
    rum = predicted_rating(u,m,t, t_mean_u)
    prob_sum = 1.0 / (1.0 + np.exp(-s*(c*rum - b)))
    return prob_sum

def sample_multinomial(w):
    """
    Returns the index of a sample from a multinomial distribution
    """
    x = np.random.uniform(0,1)
    for i,v in enumerate(np.cumsum(w)):
        if x < v: return i
    return len(w)-1

def sample_multiple_indices(p):
    """
    Samples indices from a joint probability distribution
    """
    (Y, Z, S) = p.shape
    dist = list()
    for y in range(Y):
        for z in range(Z):
            for s in range(S):
                dist.append(p[y,z,s])
    index = sample_multinomial(dist)
    y = index // (Z * S)                 #Get indices of matrix from the list indices.
    rem = index % (Z * S)
    z = rem // S
    s = rem % S
    return (y, z, s)

class GibbsSampler:
    """
    Class to handle Gibbs Sampling
    """
    def __init__(self, vocab_size, review_matrix, rating_list, movie_dict, user_dict, movie_reviews, word_dictionary,
                 U, M, R, test_indices):
        """
        Constructor
        """
        self.Y = Y
        self.Z = A
        self.S = S
        self.M = M
        self.U = U
        self.R = R
        self.A = A

        self.vocab_size = vocab_size
        self.n_reviews = len(review_matrix)

        # Number of times y occurs
        self.cy = np.zeros(self.Y)
        self.c = 0
        # Number of times y occurs with w
        self.cyw = np.zeros((self.Y, self.vocab_size))
        # Number of times y occurs with s and w
        self.cysw = np.zeros((self.Y, self.S, self.vocab_size))
        # Number of times y occurs with s
        self.cys = np.zeros((self.Y, self.S))
        # Number of times y occurs with z and w
        self.cyzw = np.zeros((self.Y, self.Z, self.vocab_size))
        # Number of times y occurs with z
        self.cyz = np.zeros((self.Y, self.Z))
        # Number of times y occurs with m and w
        self.cmyw = np.array([csr_matrix((self.Y, self.vocab_size), dtype=np.float) for _ in range(self.M)])
        # Number of times y occurs with m
        self.cym = np.zeros((self.Y, self.M))
        # map for sentiment values
        self.senti_map = [1,-1]


        self.Nums = np.zeros((self.R,self.S))
        self.Numas = np.zeros((self.R,self.A,self.S))
        self.Numa = np.zeros((self.R,self.A))

        self.topics = {}

        for movie in range(self.M):
            for (rev,r) in movie_reviews[movie]:
                if r in test_indices:
                    continue
                for i, word in enumerate(rev.strip().split()):
                    w = word_dictionary[word]
                    # Choose a random assignment of y, z, w
                    (y, z, s) = (np.random.randint(self.Y), np.random.randint(self.Z), np.random.randint(self.S))
                    # Assign new values
                    self.cy[y] += 1
                    self.c += 1
                    self.cyw[y,w] += 1
                    self.cysw[y,s,w] += 1
                    self.cys[y,s] += 1
                    self.cyzw[y,z,w] += 1
                    self.cyz[y,z] += 1
                    self.cmyw[movie][y,w] += 1
                    self.cym[y, movie] += 1
    
                    if y == 1:
                        self.Nums[r,s] +=1
                    if y == 2:
                        self.Numas[r,z,s] +=1
                    if y == 3:
                        self.Numa[r,z] +=1
    
                    self.topics[(r, i)] = (y, z, s)
    
    def assignment(self, x):
    
        global alpha_vu
        global v_u
        
        global alpha_bu
        global b_u
        
        global alpha_tu
        global theta_u
        
        global v_m
        global b_m
        global theta_m
        
        global M_a
        global b_o
        
        prev_min = 0; prev_max = self.U*K
        alpha_vu = x[prev_min:prev_max].reshape((self.U,K), order='F')
        
        prev_min = prev_max; prev_max += self.U*K 
        v_u = x[prev_min:prev_max].reshape((self.U,K), order='F')
    
        prev_min = prev_max; prev_max += self.U
        alpha_bu = x[prev_min:prev_max].reshape((self.U,1), order='F')
        
        prev_min = prev_max; prev_max += self.U
        b_u = x[prev_min:prev_max].reshape((self.U,1), order='F')
        
        prev_min = prev_max; prev_max += self.U*A
        alpha_tu = x[prev_min:prev_max].reshape((self.U,A), order='F')
        
        prev_min = prev_max; prev_max += self.U*A
        theta_u = x[prev_min:prev_max].reshape((self.U,A), order='F')
        
        prev_min = prev_max; prev_max += self.M*K
        v_m = x[prev_min:prev_max].reshape((self.M,K), order='F')
        
        prev_min = prev_max; prev_max += self.M*1
        b_m = x[prev_min:prev_max].reshape((self.M,1), order='F')
        
        prev_min = prev_max; prev_max += self.M*A
        theta_m = x[prev_min:prev_max].reshape((self.M,A), order='F')
        
        prev_min = prev_max; prev_max += A*K
        M_a = x[prev_min:prev_max].reshape((A,K), order='F')
        
        b_o = x[-1]


    def _conditional_distribution(self, u, w, m, t, t_mean_u):
        """
        Returns the CPD for word w in the review by user u for movie m
        """
        p_z = np.zeros((self.Y, self.Z, self.S))
        
        # y = 0
        p_z[0,0,0] = (self.cy[0] + gamma) / (self.c + 5 * gamma)
        p_z[0,0,0] = (p_z[0,0,0] * (self.cyw[0,w] + eta)) / (self.cy[0] + eta)


        # y = 1
        for s in range(self.S):
            p_z[1,0,s] = (self.cy[1] + gamma) / (self.c + 5 * gamma)
            p_z[1,0,s] = (p_z[1,0,s] * (self.cysw[1,s,w] + eta)) / (self.cys[1,s] + eta)
            p_z[1,0,s] = p_z[1,0,s] * aggregate_sentiment_probability(self.senti_map[s],u,m,t, t_mean_u)

        # y = 2
        for z in range(self.Z):
            for s in range(self.S):
                p_z[2,z,s] = (self.cy[2] + gamma) / (self.c + 5 * gamma)
                p_z[2,z,s] = (p_z[2,z,s] * (self.cyzw[2,z,w] + eta)) / (self.cyz[2,z] + eta)
                p_z[2,z,s] = p_z[2,z,s] * (joint_aspect(u, m,t, t_mean_u)[z])
                p_z[2,z,s] = p_z[2,z,s] * aspect_sentiment_probability(self.senti_map[s],u,m,z,t, t_mean_u)

        # y = 3
        for z in range(self.Z):
            p_z[3,z,0] = (self.cy[3] + gamma) / (self.c + 5 * gamma)
            p_z[3,z,0] = (p_z[3,z,0] * (self.cyzw[3,z,w] + eta)) / (self.cyz[3,z] + eta)
            p_z[3,z,0] = p_z[3,z,0] * (joint_aspect(u,m,t, t_mean_u)[z])

        # y = 4
        p_z[4,0,0] = (self.cy[4] + gamma) / (self.c + 5 * gamma)
        p_z[4,0,0] = (p_z[4,0,0] * (self.cmyw[m][4,w] + eta)) / (self.cym[4,m] + eta)


        # Normalize
        p_z = p_z / p_z.sum()

        return p_z


    def run(self, vocab_size, review_matrix, rating_list, user_dict, movie_dict, movie_reviews, word_dictionary,t_mean, params, test_indices, path_to_save_results, max_iter=1):
        """
        Perform sampling max_iter times
        """
        
        self.assignment(params)

        for it in range(max_iter):
            print('Gibbs Sampling Iteration: %d' % it)
            
            for movie in range(self.M):    
                for (rev,r) in movie_reviews[movie]:
                    if r in test_indices:
                        continue
                    for i, word in enumerate(rev.strip().split()):
                        w = word_dictionary[word]
                        (y, z, s) = self.topics[(r, i)]
                        
                        # Exclude current assignment
                        self.cy[y] -= 1 # specific to y
                        self.c -= 1     # sum over all y
                        self.cyw[y,w] -= 1
                        self.cysw[y,s,w] -= 1
                        self.cys[y,s] -= 1
                        self.cyzw[y,z,w] -= 1
                        self.cyz[y,z] -= 1
                        
                        self.cmyw[movie][y,w] -= 1
                        self.cym[y, movie] -= 1
    
                        # Get next distribution
                        
                        u = rating_list[r]['u']
                        t = rating_list[r]['t']

                        if y == 1:
                            self.Nums[r,s] -=1
                        if y == 2:
                            self.Numas[r,z,s] -=1
                        if y == 3:
                            self.Numa[r,z] -=1
                        
                        p_z = self._conditional_distribution(u, w, movie, t, t_mean[u]) # Eq. 13 for all values of y,z,s -> computing tensor
                        (y, z, s) = sample_multiple_indices(p_z)
    
                        # Assign new values
                        self.cy[y] += 1
                        self.c += 1
                        self.cyw[y,w] += 1
                        self.cysw[y,s,w] += 1
                        self.cys[y,s] += 1
                        self.cyzw[y,z,w] += 1
                        self.cyz[y,z] += 1
                        
                        self.cmyw[movie][y,w] += 1
                        self.cym[y, movie] += 1
                        
                        if y == 1:
                            self.Nums[r,s] +=1
                        if y == 2:
                            self.Numas[r,z,s] +=1
                        if y == 3:
                            self.Numa[r,z] +=1
                        
                        self.topics[(r, i)] = (y, z, s)
                save_npz(path_to_save_results + 'movie_aspects/%s'%(movie), self.cmyw[movie])
            ##saving files for analysis
            np.save(path_to_save_results + 'cyw.npy', self.cyw)
            np.save(path_to_save_results + 'cyzw.npy', self.cyzw)
            np.save(path_to_save_results + 'cysw.npy', self.cysw)
            np.save(path_to_save_results + 'Nums.npy', self.Nums)
            np.save(path_to_save_results + 'Numas.npy', self.Numas)
            np.save(path_to_save_results + 'Numa.npy', self.Numa)
            return (self.Nums, self.Numas, self.Numa)
        
