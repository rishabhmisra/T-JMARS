from constants import *
import math
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import numpy.matlib


def dev_t(t, tu_mean):
    return np.sign(t-tu_mean)*abs(t-tu_mean)**beta
#    return 0.0

def assign_params(x,U,M,R):
    prev_min = 0; prev_max = U*K
    alpha_vu = x[prev_min:prev_max].reshape((U,K), order='F')
    
    prev_min = prev_max; prev_max += U*K 
    v_u = x[prev_min:prev_max].reshape((U,K), order='F')

    prev_min = prev_max; prev_max += U
    alpha_bu = x[prev_min:prev_max].reshape((U,1), order='F')
    
    prev_min = prev_max; prev_max += U
    b_u = x[prev_min:prev_max].reshape((U,1), order='F')
    
    prev_min = prev_max; prev_max += U*A
    alpha_tu = x[prev_min:prev_max].reshape((U,A), order='F')
    
    prev_min = prev_max; prev_max += U*A
    theta_u = x[prev_min:prev_max].reshape((U,A), order='F')
    
    prev_min = prev_max; prev_max += M*K
    v_m = x[prev_min:prev_max].reshape((M,K), order='F')
    
    prev_min = prev_max; prev_max += M*1
    b_m = x[prev_min:prev_max].reshape((M,1), order='F')
    
    prev_min = prev_max; prev_max += M*A
    theta_m = x[prev_min:prev_max].reshape((M,A), order='F')
    
    prev_min = prev_max; prev_max += A*K
    M_a = x[prev_min:prev_max].reshape((A,K), order='F')
    
    b_o = x[-1]
    
    return (alpha_vu, v_u, alpha_bu, b_u, alpha_tu, theta_u, v_m, b_m, theta_m, M_a, b_o)


def calculate_rmse(x,U,M,t_mean,rating_list,test_indices):
    
    (alpha_vu, v_u, alpha_bu, b_u, alpha_tu, theta_u, v_m, b_m, theta_m, M_a, b_o) = assign_params(x,U,M,len(rating_list))
    
    num_theta_uma = np.zeros((len(rating_list), A)) 
    for i in range(len(rating_list)): 
        m = rating_list[i]['m']
        u = rating_list[i]['u']
        t = rating_list[i]['t']
        num_theta_uma[i] = np.exp(theta_u[u] + dev_t(t, t_mean[u])*alpha_tu[u] + theta_m[m])
        
    
    theta_uma = np.divide(num_theta_uma.T,num_theta_uma.sum(axis=1)).T

    M_sum = np.dot(theta_uma, M_a)
    RMSE_train = 0
    RMSE_test = 0
    for i in range(len(rating_list)):
        m = rating_list[i]['m']
        u = rating_list[i]['u']
        t = rating_list[i]['t']
        r = rating_list[i]['r']

        v_ut = v_u[u] + dev_t(t, t_mean[u])*alpha_vu[u]
        b_ut = b_u[u] + dev_t(t, t_mean[u])*alpha_bu[u]
        r_hat =  np.dot(np.dot(v_ut, np.diag(M_sum[i])), v_m[m].T) + b_o + b_ut + b_m[m]
        if i not in test_indices:
            RMSE_train += (r - r_hat)**2
        else:
            RMSE_test += (r - r_hat)**2
        
    return (math.sqrt(RMSE_train/(len(rating_list) - len(test_indices))),
            math.sqrt(RMSE_test/(len(test_indices))))


def func(params, args):
    """
    Computes the value of the objective function required for gradient descent
    """
    
    Nums = args[0]
    Numas = args[1]
    Numa = args[2]
    rating_list = args[3]
    t_mean = args[4]
    U = args[5]; M = args[6]; R = args[7]
    test_indices = args[8]
    lamda = args[9]
    
    (alpha_vu, v_u, alpha_bu, b_u, alpha_tu, theta_u, v_m, b_m, theta_m, M_a, b_o) = assign_params(params,U,M,R)

    r_hat = np.zeros(R)
    num_theta_uma = np.zeros((len(rating_list), A)) 

    for i in range(len(rating_list)):
        
        m = rating_list[i]['m']
        u = rating_list[i]['u']
        t = rating_list[i]['t']
        num_theta_uma[i] = np.exp(theta_u[u] + dev_t(t, t_mean[u])*alpha_tu[u] + theta_m[m])
        
    
    theta_uma = np.divide(num_theta_uma.T,num_theta_uma.sum(axis=1)).T

    M_sum = np.dot(theta_uma, M_a)
    
    loss1 = 0
    loss3 = 0

    for i in range(len(rating_list)):
        if i in test_indices:
            continue
        
        m = rating_list[i]['m']
        u = rating_list[i]['u']
        t = rating_list[i]['t']
        r = rating_list[i]['r']

        v_ut = v_u[u] + dev_t(t, t_mean[u])*alpha_vu[u]
        b_ut = b_u[u] + dev_t(t, t_mean[u])*alpha_bu[u]
        r_hat[i] =  np.dot(np.dot(v_ut, np.diag(M_sum[i])), v_m[m].T) + b_o + b_ut + b_m[m]
        loss1 += epsilon*(r - r_hat[i])**2
        
        for j in range(A):
            ruma = np.dot(np.dot(v_ut, np.diag(M_a[j])), v_m[m].T) + b_o + b_ut + b_m[m]
            # sentiment - 0 -> positive,  1-> negative
            loss3 += np.multiply(Numas[i,j,0], np.log(1/(1 + np.exp(-1*(c*ruma - b))))) + np.multiply(Numas[i,j,1], np.log(1/(1 + np.exp((c*ruma - b)))))
    
    loss2 = np.multiply(Nums[:,0], np.log(1/(1 + np.exp(-1*(c*r_hat - b))))) + np.multiply(Nums[:,1], np.log(1/(1 + np.exp((c*r_hat - b)))))    
    loss4 = (np.multiply(Numa, np.log(theta_uma))).sum()
    total_loss = loss1.sum() - loss2.sum() - loss3.sum() - loss4.sum()

    return total_loss + lamda*np.linalg.norm(params, ord=1)


def fprime(params, args):

    Nums = args[0]
    Numas = args[1]
    Numa = args[2]
    rating_list = args[3]
    t_mean = args[4]
    U = args[5]; M = args[6]; R = args[7]
    test_indices = args[8]
    lamda = args[9]
    
    (alpha_vu, v_u, alpha_bu, b_u, alpha_tu, theta_u, v_m, b_m, theta_m, M_a, b_o) = assign_params(params,U,M,R)

    num_theta_uma = np.zeros((len(rating_list), A))

    for i in range(len(rating_list)):
        m = rating_list[i]['m']
        u = rating_list[i]['u']
        t = rating_list[i]['t']

        num_theta_uma[i] = np.exp(theta_u[u] + dev_t(t, t_mean[u])*alpha_tu[u] + theta_m[m])

    theta_uma = np.divide(num_theta_uma.T,num_theta_uma.sum(axis=1)).T
    M_sum = np.dot(theta_uma, M_a)

# MAde changes here.

    final_grad_vu = np.zeros((U,K))
    final_grad_alpha_vu = np.zeros((U,K))
        
    final_grad_bu = np.zeros((U,1))
    final_grad_alpha_bu = np.zeros((U,1))

    final_grad_thetau = np.zeros((U,A))
    final_grad_alpha_thetau = np.zeros((U,A))

    final_grad_vm = np.zeros((M,K))
    final_grad_b_m = np.zeros((M,1))     
    final_grad_theta_m = np.zeros((M,A))
        
    final_grad_M_a = np.zeros((A,K))
    final_grad_bo = 0.


    for i in range(len(rating_list)):
        if i in test_indices:
            continue
        m = rating_list[i]['m']
        u = rating_list[i]['u']
        t = rating_list[i]['t'] 
        r = rating_list[i]['r']
        
        v_ut = v_u[u] + dev_t(t, t_mean[u])*alpha_vu[u]
        b_ut = b_u[u] + dev_t(t, t_mean[u])*alpha_bu[u]
        r_hat = np.dot(np.dot(v_ut, np.diag(M_sum[i])), v_m[m].T) + b_o + b_ut + b_m[m]
        rating_error = 2*epsilon*(r_hat - r)
        
        ########## For term A

        gradA_vu = np.multiply(M_sum[i], v_m[m].T)
        gradA_alpha_vu = gradA_vu*dev_t(t, t_mean[u])
        
        gradA_bu = 1
        gradA_alpha_bu = dev_t(t, t_mean[u])

        gradA_theta = np.multiply( np.dot( np.multiply(v_ut, M_a), v_m[m]), (theta_uma[i] * (1 - theta_uma[i]))) #revisit
        gradA_thetau = gradA_theta
        gradA_alpha_thetau = gradA_thetau*dev_t(t, t_mean[u])

        gradA_vm = np.multiply(v_ut, M_sum[i])
        gradA_b_m = 1        
        gradA_theta_m = gradA_theta

        gradA_M_a = np.dot(np.matrix(theta_uma[i]).T, np.matrix(np.multiply(v_ut, v_m[m])))
        gradA_bo = 1 

        ########### For term B
        
        gradB_factor =  (Nums[i,0] /(1 + np.exp(c*r_hat - b)))*1*c + (Nums[i,1]/(1 + np.exp(-1*(c*r_hat - b))))*(-1)*c

        ########### For term C
        gradC_vu = np.zeros(K)
        gradC_alpha_vu = np.zeros(K)
        gradC_bu = 0 
        gradC_alpha_bu = 0
        gradC_vm = np.zeros(K)
        gradC_b_m = 0
        gradC_M_a = np.zeros((A,K))
        gradC_bo = 0

        for j in range(A):
            ruma = np.dot(np.multiply(v_ut, M_a[j]), v_m[m].T) + b_o + b_ut + b_m[m]
            gradC_factor = (Numas[i,j,0]/(1 + np.exp(c*ruma - b)))*1*c + (Numas[i,j,1]/(1 + np.exp(-1*(c*ruma - b))))*(-1)*c

            gradC_vu += gradC_factor*np.multiply(M_a[j], v_m[m])
            
            gradC_bu += gradC_factor
            gradC_alpha_bu += gradC_factor*dev_t(t, t_mean[u])

            gradC_vm += gradC_factor*np.multiply(M_a[j], v_ut)
            gradC_b_m += gradC_factor        
            
            gradC_M_a[j] = gradC_factor*np.matrix(np.multiply(v_ut, v_m[m]))
            gradC_bo += gradC_factor

        gradC_alpha_vu = gradC_vu*dev_t(t, t_mean[u])
        
        ########### For term D
        
        softmax_matrix = np.zeros((A,A))
        numa_by_theta_uma = np.zeros(A)

        for j in range(A):
            numa_by_theta_uma[j] = Numa[i,j]/theta_uma[i][j]
            for k in range(A):
                if j == k:
                    softmax_matrix[j][k] = theta_uma[i][j]*(1-theta_uma[i][k])
                else:
                    softmax_matrix[j][k] = -1*theta_uma[i][j]*theta_uma[i][k]

        gradD_thetau = np.dot(numa_by_theta_uma, softmax_matrix.T)
        gradD_thetam = np.dot(numa_by_theta_uma, softmax_matrix.T)
        gradD_alpha_thetau = gradD_thetau*dev_t(t, t_mean[u])

        ############### Final

        final_grad_vu[u] += (rating_error - gradB_factor)*gradA_vu - gradC_vu
        final_grad_alpha_vu[u] += (rating_error - gradB_factor)*gradA_alpha_vu - gradC_alpha_vu
        
        final_grad_bu[u] += (rating_error - gradB_factor)*gradA_bu - gradC_bu
        final_grad_alpha_bu[u] += (rating_error - gradB_factor)*gradA_alpha_bu - gradC_alpha_bu

        final_grad_thetau[u] += (rating_error - gradB_factor)*gradA_thetau - gradD_thetau
        final_grad_alpha_thetau[u] += (rating_error - gradB_factor)*gradA_alpha_thetau - gradD_alpha_thetau

        final_grad_vm[m] += (rating_error - gradB_factor)*gradA_vm - gradC_vm
        final_grad_b_m[m] += (rating_error - gradB_factor)*gradA_b_m - gradC_b_m      
        final_grad_theta_m[m] += (rating_error - gradB_factor)*gradA_theta_m - gradD_thetam

        final_grad_M_a += np.multiply((rating_error - gradB_factor),gradA_M_a) - gradC_M_a
        
        final_grad_bo += (rating_error - gradB_factor)*gradA_bo - gradC_bo

    return numpy.concatenate(((final_grad_alpha_vu + lamda*np.sign(alpha_vu)).flatten('F'), 
            (final_grad_vu + lamda*np.sign(v_u)).flatten('F'), 
            (final_grad_alpha_bu + lamda*np.sign(alpha_bu)).flatten('F'), 
            (final_grad_bu + lamda*np.sign(b_u)).flatten('F'), 
            (final_grad_alpha_thetau + lamda*np.sign(alpha_tu)).flatten('F'), 
            (final_grad_thetau + lamda*np.sign(theta_u)).flatten('F'), 
            (final_grad_vm + lamda*np.sign(v_m)).flatten('F'), 
            (final_grad_b_m  + lamda*np.sign(b_m)).flatten('F'), 
            (final_grad_theta_m + lamda*np.sign(theta_m)).flatten('F'), 
            (final_grad_M_a + lamda*np.sign(M_a)).flatten('F'),
            (np.array([final_grad_bo]) + lamda*np.sign(np.array([b_o]))).flatten('F')))


def optimizer(Nums,Numas,Numa,rating_list,t_mean, params,U,M,R,test_indices,save_test_rmse):
    """
    Computes the optimal values for the parameters required by the JMARS model using lbfgs
    """
    
#    print('opti', params[:10])
    lamda = 0.0  # 0.1 - 0.9677
    args = [Nums,Numas,Numa,rating_list,t_mean,U,M,R,test_indices,lamda]
    # e = 0.001
    # sav = []
    # grad = fprime(params,args)
    # np.save('mini_grad_fprime.npy',grad)
    # f_base = func(params, args)
    # for i in range(len(params)):
    #     new_params = np.copy(params) 
    #     new_params[i] += e
    #     grad_num = (func(new_params, args) - f_base)/e
    #     if i%1000 == 0:
    #         print(i)
    #     sav.append(abs(grad_num - grad[i]))

    # np.save('mini_grad_diff.npy',sav)
    # print(max(sav))
    learning_rate = 0.00001
    for i in range(10):
        params -= learning_rate*fprime(params,args)
        train_rmse, test_rmse = calculate_rmse(params,U,M,t_mean,rating_list,test_indices)
        save_test_rmse.append((train_rmse, test_rmse))
        print ('Loss: ' + str(func(params, args)) + '------------' + 'RMSE ', train_rmse, test_rmse)
   # params,l,_ = fmin_l_bfgs_b(func, x0=params, fprime=fprime, args=args, approx_grad=False, maxfun=1, maxiter=10)
   # print ('Loss: ' + str(l) + '------------' + 'RMSE ' + str(calculate_rmse(params,U,M,t_mean,rating_list)))

# RMSprop  
#    gamma = 0.9
#    eps = 0.00000001
#    lamda = 0.0
#    
#    args = [Nums,Numas,Numa,rating_list,t_mean,U,M,R,test_indices,lamda]
#    print ()
#    for i in range(10):
#        grad = fprime(params,args)
#        cache = gamma*cache + (1-gamma)*(grad**2)
#        params -= learning_rate * grad / (np.sqrt(cache + eps))
#        print ('Loss: ' + str(func(params, args)) + '------------' + 'RMSE ' + str(calculate_rmse(params,U,M,t_mean,rating_list,test_indices)))

    return params,save_test_rmse