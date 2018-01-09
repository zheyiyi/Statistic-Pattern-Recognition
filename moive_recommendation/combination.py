# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 22:30:06 2017

@author: zheyiyi
"""




import os
import numpy as np
import math
import sys


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_data(filename):
    results = []
    #path = os.path.join("/Users","zheyiyi","Desktop" ,"moveData", "ml-100k", filename)
    f = open(filename,"r")
    results = [line.split()[0:-1] for line in f]
    results = np.array(results, dtype = "float")
    user_numbers = int(np.max(results[:,0]))
    movie_numbers = int(np.max(results[:,1]))
    
    test_length = int(len(results) * 0.9)
    train_datas = results[:test_length]
    test_datas = results[test_length:]
    train_datas = train_datas[train_datas[:, 0].argsort()]  
    f.close()
    return train_datas, test_datas, user_numbers, movie_numbers

def cal_mean(train_datas):
    num = len(train_datas)    
    return np.sum(train_datas[:,2]) / num
    
def initial_bias(train_datas, user_numbers, movie_numbers, mean):
    bu = {}
    bi = {}
    bu_num = {}
    bi_num = {}
    index1 = 0
    index2 = 0
    
    
    for i in range(user_numbers):
        index1 = 0
        while index1 < len(train_datas) and train_datas[index1][0] == i + 1:
            j = int(train_datas[index1][1])
            
            if j not in bi:
                bi[j] = 0
                bi_num[j] = 0
            bi[j] += train_datas[index1][2] - mean
            
            bi_num[j] += 1
            
            index1 +=1
           
    for i in range(movie_numbers):
        if i not in bi_num:
            bi[i] = 0
        elif bi_num[i] >= 1:
            
            bi[i] = bi[i] / (bi_num[i]+25)
           
      
            
    
    for i in range(user_numbers):
        index2 = 0

        while index2 < len(train_datas) and train_datas[index2][0] == i + 1:
            j = int(train_datas[index2][1])
            if i not in bu:
                bu[i] = 0
                bu_num[i] =0
                
            
            bu[i] += train_datas[index2][2] - mean - bi[j]
            
            bu_num[i] += 1
            index2 += 1
  
    
    for i in range( movie_numbers):
        if i not in bu_num:
            bu[i] = 0
        elif bu_num[i] >= 1:
            bu[i] = bu[i] / (bu_num[i]+ 10)
        
    return bu,bi   
    
def svd_train(filename, rank):
    train_datas, test_datas, user_numbers, movie_numbers = load_data(filename)
    mean = cal_mean(train_datas)
    
    bu, bi = initial_bias(train_datas, user_numbers, movie_numbers, mean)
    gama = 0.02
    lamda = 0.3
    step = 100
    slowrate = 0.99
    min_value = sys.maxsize
    u = np.random.randn(user_numbers, rank)
    v = np.random.randn(movie_numbers, rank)
    print(predictRMSE(test_datas,u, v,mean, bu, bi))
    
    for z in range(step):
        index = 0
        for i in range(user_numbers):
            
            while index < len(train_datas) and train_datas[index][0] == i + 1:
                j = int(train_datas[index][1] - 1)
              
                pui = (mean + bu[i] + bi[j]) + np.dot(u[i],v[j].T)
                eui = train_datas[index][2] - pui
                bu[i] += gama * (eui - lamda * bu[i])
                bi[j] += gama * (eui - lamda * bi[j])
                u[i] += gama * (eui * v[j] - lamda * u[i])
                v[i] += gama * (eui * u[i] - lamda * v[j])
                index += 1
            
        gama *= slowrate
        value = predictRMSE(test_datas,u, v,mean, bu, bi)
        if min_value > value:
           min_value = value
        else:
            break
        print(predictRMSE(test_datas,u, v,mean, bu, bi))
    return min_value, u, v, bu, bi,mean

def predictRMSE(test_ratings, u, v,mean, bu, bi):
    # predict movie ratings on test set and calculate RMSE
    RMSE = 0
    for rating in test_ratings:
        pui = mean + bu[int(rating[0] - 1)] + bi[int(rating[1] - 1)] + np.dot(u[int(rating[0] - 1), :], v[int(rating[1] - 1), :])
        RMSE += np.square(rating[2] - pui)
    RMSE = math.sqrt(RMSE / len(test_ratings))
    return RMSE


def predict_rmse(test_datas, u, v):
    # predict movie ratings on test set by calculating rmse
    rmse = 0
    for vaule in test_datas:
        rmse += np.square(vaule[2] - np.dot(u[int(vaule[0] - 1)], v[int(vaule[1] - 1)]))
    rmse = np.sqrt(rmse / len(test_datas))
    return rmse


def train(filename, rank):
    print("finalproject")
    # Training of the variational inference matrix factorization model
    # Load data and split
    train_datas,test_datas, user_numbers, movie_numbers = load_data(filename)
    n = len(train_datas)
    # Initialize model parameters, the variances
    # The variances of V matrix, rhos, are held constant
    # The variances of U matrix, sigma square, are placed on the diagonal of a diagonal matrix, sigmasq_matrix
    tau = 1
    sigma_matrix =  np.identity(rank)
    # Initialize U and V randomly from N(0, 1), normal distribution with mean 0 and variance 1
    U = np.random.randn(user_numbers, rank)
    V = np.random.randn(movie_numbers, rank)
    
    psi = np.zeros((movie_numbers, rank, rank))
    
    new_predict = 0
    old_predict = predict_rmse(test_datas, U, V)
    
    s_list =[np.identity(rank) * (1 / rank) for i in range(movie_numbers)]
    t_list = [np.zeros(rank) for j in range(movie_numbers)]
    new_sigma = np.zeros((rank, rank))
    
    while abs(old_predict - new_predict) > 0.001:
        print(new_predict, old_predict)
        old_predict = new_predict 
        
        new_tau = 0   
        index_1 = 0
        index_2 = 0
        # update Q(u)
        for i in range(user_numbers):
            # Update Phi_i and u_i
            phi = sigma_matrix
            u_mean = np.zeros(rank)
            while index_1 < n and train_datas[index_1][0] == i + 1:
                j = int(train_datas[index_1][1] - 1)   
                a = np.array([V[j]])  
                phi = phi + (psi[j] + np.dot(a.T, a)) / tau
                u_mean = u_mean +  train_datas[index_1][2] * V[j] / tau
                index_1 = index_1 + 1
                
            phi = np.linalg.inv(phi)     # phi - （5，5） u_mean - (5,)
            u_mean = np.dot(phi, u_mean)
           
            U[i] = u_mean
            # Update S_j and t_j, add value to new_tausq for tausq update
            while index_2 < n and train_datas[index_2][0] == i + 1:
                j = int(train_datas[index_2][1] - 1)
                b = np.array([u_mean])
                s_list[j] = s_list[j] + (phi + np.dot(b.T, b)) / tau
                t_list[j] = t_list[j] + train_datas[index_2][2] * u_mean / tau
                a = np.array([V[j]])     
                trace_value = np.sum((phi + np.outer(u_mean, u_mean)) * (psi[j] + np.dot(a.T, a)))
                new_tau = new_tau + train_datas[index_2][2]**2 - 2 * train_datas[index_2][2] * np.dot(U[i], V[j]) + trace_value
                index_2 = index_2 + 1

            # Add value from Phi and u_mean to new_sigmasq
            for z in range(rank):
                new_sigma[z, z] = new_sigma[z, z] + phi[z, z] + np.square(u_mean[z])

        # Update Psi_j and v_j
        for j in range(movie_numbers):
            psi[j] = np.linalg.inv(s_list[j])
           
            V[j] = np.dot(psi[j], t_list[j])

        # Update the variances, update sigmasq
        sigma_matrix = 1 /(user_numbers -1) * new_sigma
        # Update tausq
        tau = 1 /(n - 1) * new_tau
        new_predict = predict_rmse(test_datas, U, V)
        
    return new_predict, U, V,test_datas
    
new_predict, u, v,test_set = train("u.data", 3)
min_value,u_svd,v_svd,bu,bi,mean = svd_train("u.data", 1)

print(test_set)
  

def combination_rmse(test_data, u1, v1,u2,v2,bu, bi, mean):
    # predict movie ratings on test set by calculating rmse
  
    rmse = 0
    for datas in test_data:
        pui = mean + bu[int(datas[0] - 1)] + bi[int(datas[1] - 1)] + np.dot(u2[int(datas[0] - 1), :], v2[int(datas[1] - 1), :])
        rmse += np.square(datas[2] - (np.dot(u1[int(datas[0] - 1)], v1[int(datas[1] - 1)]) + pui) /2)
    rmse = np.sqrt(rmse / len(test_data))
    return rmse

value = combination_rmse(test_set, u, v, u_svd, v_svd, bu, bi, mean)
print(new_predict, min_value, value )