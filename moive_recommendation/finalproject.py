# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 14:07:41 2017

@author: zheyiyi
"""

import numpy as np
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_data(filename):
    results = []
    #path = os.path.join("/Users","zheyiyi","Desktop" ,"moveData", filename1, filename2)
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
    

def predict_rmse(test_datas, u, v):
    # predict movie ratings on test set by calculating rmse
    rmse = 0
    for vaule in test_datas:
        rmse += np.square(vaule[2] - np.dot(u[int(vaule[0] - 1)], v[int(vaule[1] - 1)]))
    rmse = np.sqrt(rmse / len(test_datas))
    return rmse


def train(filename, rank):
    print("finalproject")
    
    train_datas,test_datas, user_numbers, movie_numbers = load_data(filename)
    n = len(train_datas)
     
    tau = 1
    sigma_matrix =  np.identity(rank)
    U = np.random.randn(user_numbers, rank)
    V = np.random.randn(movie_numbers, rank)
    
    psi = np.zeros((movie_numbers, rank, rank))
    
    new_predict = 0
    old_predict = predict_rmse(test_datas, U, V)
    
    s_list =[np.identity(rank) * (1 / rank) for i in range(movie_numbers)]
    t_list = [np.zeros(rank) for j in range(movie_numbers)]
    new_sigma = np.zeros((rank, rank))
    
    while abs(old_predict - new_predict) > 0.001:
       
        old_predict = new_predict 
        
        new_tau = 0   
        index_1 = 0
        index_2 = 0
    
        for i in range(user_numbers):
      
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
            while index_2 < n and train_datas[index_2][0] == i + 1:
                j = int(train_datas[index_2][1] - 1)
                b = np.array([u_mean])
                s_list[j] = s_list[j] + (phi + np.dot(b.T, b)) / tau
                t_list[j] = t_list[j] + train_datas[index_2][2] * u_mean / tau
                a = np.array([V[j]])     
                trace_value = np.sum((phi + np.outer(u_mean, u_mean)) * (psi[j] + np.dot(a.T, a)))
                new_tau = new_tau + train_datas[index_2][2]**2 - 2 * train_datas[index_2][2] * np.dot(U[i], V[j]) + trace_value
                index_2 = index_2 + 1

            for z in range(rank):
                new_sigma[z, z] = new_sigma[z, z] + phi[z, z] + np.square(u_mean[z])

        
        for j in range(movie_numbers):
            psi[j] = np.linalg.inv(s_list[j])
           
            V[j] = np.dot(psi[j], t_list[j])

        sigma_matrix = 1 /(user_numbers -1) * new_sigma
        
        tau = 1 /(n - 1) * new_tau
        new_predict = predict_rmse(test_datas, U, V)
        
    return new_predict, U, V
"""
result_100k = []
result_1m = []
rank = []
for i in range(1,16): 
    new_predict, u, v = train("ml-100k","u.data", i)
    result_100k.append(new_predict)
   # new_predict, u, v = train("ml-1m","ratings.dat", i)
   # result_1m.append(new_predict)
    rank.append(i)
 
plt.figure()

print(result_100k)
#print(result_1m)
plt.plot(rank, result_100k, 'r-',label ="the rmse of move_100k svd as a function of rank")
#plt.plot(rank, result_1m, 'k-',label ="the rmse of move_1m svd as a function of rank")

plt.xlim(0,17)
plt.ylim(0,1.5)
plt.legend()
plt.title("Figure 1")
plt.ylabel("rmse")
plt.xlabel("rank size")
plt.savefig('graph1.png')
"""
