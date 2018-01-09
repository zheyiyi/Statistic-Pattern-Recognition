# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 01:20:34 2017

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
    #path = os.path.join(filename)
    f = open(filename,"r")
    
    results = [line.split()[0:-1] for line in f]

    results = np.array(results, dtype = "float")
    user_numbers = int(np.max(results[:,0]))
    movie_numbers = int(np.max(results[:,1]))
    
    test_length = int(len(results) * 0.9)
    train_datas = results[:test_length]
    test_datas = results[test_length:]
    train_datas = train_datas[train_datas[:, 0].argsort()]  
    
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
  
    
    for i in range(movie_numbers):
        if i not in bu_num:
            bu[i] = 0
        elif bu_num[i] >= 1:
            bu[i] = bu[i] / (bu_num[i]+ 10)
        
    return bu,bi   
    
def svd_train(filename,rank):
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
    
    
    for z in range(step):
        index = 0
        for i in range(user_numbers):
            
            while index < len(train_datas) and train_datas[index][0] == i + 1:
                j = int(train_datas[index][1] - 1)
                #print(mean)
                #print(bu[i])
                #print(bi[j])
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
    return min_value, u, v, bu, bi

def predictRMSE(test_ratings, u, v,mean, bu, bi):
    # predict movie ratings on test set and calculate RMSE
    RMSE = 0
    for rating in test_ratings:
        pui = mean + bu[int(rating[0] - 1)] + bi[int(rating[1] - 1)] + np.dot(u[int(rating[0] - 1), :], v[int(rating[1] - 1), :])
        RMSE += np.square(rating[2] - pui)
    RMSE = math.sqrt(RMSE / len(test_ratings))
    return RMSE
    
#for i in range(3,5): 
#    print(i)
#    min_value, u, v, bu, bi = svd_train("ml-1m", "ratings.dat", i) 

#u,v,bu,bi = svd_train("u.data", 10)
#train_datas, test_datas, user_numbers, movie_numbers = load_data("u.data")  
#cal_mean(train_datas)
"""
result = []
rank = []
for i in range(1,16): 
    min_value,u,v,bu,bi = svd_train("u.data", i)   
    result.append(min_value)
    rank.append(i)
 
plt.figure()

print(result)
plt.plot(rank, result, 'r-',label ="the rmse of svd as a function of rank")

plt.xlim(0,17)
plt.ylim(0,1.5)
plt.legend()
plt.title("Figure 1")
plt.ylabel("rmse")
plt.xlabel("rank size")
plt.savefig('graph1.png')
"""