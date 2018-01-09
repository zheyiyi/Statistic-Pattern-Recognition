# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 19:47:29 2017

@author: zheyiyi
"""


import sys
import math
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv

def open_matrix(filename):
    list_data = []
    with open(filename) as csvfile:
        data = csv.reader(csvfile, delimiter = ",")
        
        for row in data:
            list_data.append(row)
        return list_data

        
def train_generate(list_data,d):
    result = []
    for data in list_data:
        vector = []
        for i in range(d + 1):           
            vector.append(math.pow(float(data[0]),i))
        result.append(vector)
        
    return np.array(result,dtype = float)
 
def bastys(x,y):
    new_a = random.randrange(1,10)
    new_b = random.randrange(1,10)  
    a = 11
    b = 11
    n = len(y)
    #print(x.shape, y.shape)
    #print(a,b,new_a,new_b)
    while abs(new_a - a) > 0.0000001 and abs(new_b - b) > 0.0000001:
        a = new_a
        b = new_b
        #print(a)
        t = np.dot(x.T,x)
        #print(t.shape)
        #print(np.shape(np.identity(len(t[0]))))
        s_n_1 = np.add(a * np.identity(len(t[0])) ,b * t)
        s_n = np.linalg.inv(s_n_1)  ## see hoe to -1
        m_n = b * np.dot(np.dot(s_n, x.T), y)
        
        evige_value = np.linalg.eigvals(s_n_1)
       # print(evige_value)
        lamb_da= evige_value -a#不确定能不能这么减
        #print(evige_value)
        #print(a)
        #print(lamb_da)
       # print(len(evige_value))
        r = 0
        for lamb in lamb_da:
            r += lamb / (a + lamb)
        
        new_a = r / np.dot(m_n.T,m_n)[0][0]
        #print(new_a)
        sum = 0
        for i in range(n):
            sum += math.pow((y[i] - np.dot(m_n.T, x[i])),2)
        new_b = (n - r)/ sum

    return new_a,new_b       
   
def evidence(x,y):
    sum = 0
    a,b = bastys(x,y)
    #print(a)
    #print(x.shape)
    n,m = x.shape
    #print(len(x[0]))
    #print(len(x))
    #m = len(x[0])
    #n = len(x)
    sum += (m/2) * np.log(a)
    sum += (n/2) * np.log(b)
    
    t = np.dot(x.T,x)
    s_n = np.linalg.inv(np.add(a * np.identity(len(t[0])) ,b * t))## see hoe to -1
    m_n = b * np.dot(np.dot(s_n, x.T), y)
    #p = np.subtract(y, np.dot(x, m_n))
    #print((b/2) * (np.sqrt(np.sum(p*p, axis=0))) )
    #e_m = np.add((b/2) * (np.sqrt(np.sum(p*p, axis=0))),(a / 2) * np.dot(m_n.T, m_n)) # 有问题
    c = y - np.dot(x, m_n)
    
    e_m = (b/2.0) * (np.linalg.norm(c)**2) + (a/2.0) * np.dot(m_n.T,m_n)
    #e_m = (b/2.0) * np.dot(c.T,c) + (a/2.0) * np.dot(m_n.T,m_n)
    sum -= e_m

   
    A = a * np.identity(m) + b * np.dot(x.T,x)
    A_det = np.linalg.det(A)  
    sum -= (1/2) *  np.log(A_det)
    #print(np.log(2))
    
    sum -= (n/2) * np.log(2 * math.pi)
     
    return sum
"""
def evidence(a, b, mn, f, t):
    N, M = f.shape
    #print 'N, M ', N, M
    ft = f.T
    A = a * (np.identity(M)) + b * np.dot(ft, f)
    A_det = np.linalg.det(A)
    c = t - f.dot(mn)
    E_mn = (b/2.0) * c.dot(c) + (a/2.0) * mn.dot(mn)
    
    log_evi = M / 2.0 *(math.log(a)) + N / 2.0 *(math.log(b)) - E_mn - math.log(A_det)/2.0 - N /2.0 * (math.log(2*math.pi))
    

    return log_evi
""" 
def w_map(x,y):
    a,b = bastys(x,y)
    #print(a,b)
    t = np.dot(x.T,x)
    s_n = np.linalg.inv(a * np.identity(len(t[0])) + b * t) 
    w_max =  b * np.dot(s_n, np.dot(x.T, y))
    return w_max

def mean_squre(x,y,w):
    sum = 0
    n = len(y)
    for i in range(n):
        a = np.dot(x[i].T,w)
        sum += (a - y[i]) ** 2
    return sum / n

#f3 all evidence from degree 1 to 10
train_f3 = open_matrix("train-f3.csv")
trainr_f3 = open_matrix("trainR-f3.csv")
new_trainr_f3 = np.array(trainr_f3, dtype=float)

test_f3 = open_matrix("test-f3.csv")

testr_f3 = open_matrix("testR-f3.csv")

new_testr_f3 = np.array(testr_f3, dtype=float)
#print(new_testr_f3)
evi_list_3 = []
w_mean_list_f3 = []
w_noregular_list_f3 = []


for i in range(1,11):
    new_train_f3 = train_generate(train_f3, i)
    new_test_f3 = train_generate(test_f3, i)
    w_max_f3 = w_map(new_train_f3, new_trainr_f3)
    w_mean_squre = mean_squre(new_test_f3, new_testr_f3, w_max_f3)
    w_mean_list_f3.append(w_mean_squre)
    

    w_noregular = np.dot(np.dot(np.linalg.inv(np.dot(new_train_f3.T, new_train_f3)),new_train_f3.T),new_trainr_f3)
    w_mean_squre_no = mean_squre(new_test_f3, new_testr_f3, w_noregular)
    w_noregular_list_f3.append(w_mean_squre_no)
    
    evi_value = evidence(new_train_f3, new_trainr_f3)
    evi_list_3.append(evi_value[0])




# f5

#f3 all evidence from degree 1 to 10
train_f5 = open_matrix("train-f5.csv")
trainr_f5 = open_matrix("trainR-f5.csv")
new_trainr_f5 = np.array(trainr_f5, dtype=float)

test_f5 = open_matrix("test-f5.csv")

testr_f5 = open_matrix("testR-f5.csv")

new_testr_f5 = np.array(testr_f5, dtype=float)
#print(new_testr_f3)
evi_list_5 = []
w_mean_list_f5 = []
w_noregular_list_f5 = []


for i in range(1,11):
    new_train_f5 = train_generate(train_f5, i)
    new_test_f5 = train_generate(test_f5, i)
    w_max_f5 = w_map(new_train_f5, new_trainr_f5)
    w_mean_squre = mean_squre(new_test_f5, new_testr_f5, w_max_f5)
    w_mean_list_f5.append(w_mean_squre)
    

    w_noregular = np.dot(np.dot(np.linalg.inv(np.dot(new_train_f5.T, new_train_f5)),new_train_f5.T),new_trainr_f5)
    w_mean_squre_no = mean_squre(new_test_f5, new_testr_f5, w_noregular)
    w_noregular_list_f5.append(w_mean_squre_no)
    
    evi_value = evidence(new_train_f5, new_trainr_f5)
    evi_list_5.append(evi_value[0])
    
#print(evi_list_3[np.argmax(evi_list_3)])
#print(np.argmax(evi_list_3))
#print(w_mean_list_f3)
#print(evi_list_3)
#print(w_mean_list_f3)


d = [1,2,3,4,5,6,7,8,9,10]
plt.figure()
print("mse of noregular model for f3")
print(w_noregular_list_f3)
print("mse of bayesian model for f3")
print(w_mean_list_f3)
#print(w_noregular_list_f3)
plt.plot(d, w_noregular_list_f3, 'b-',label ="mean squre curve of no regular model of f3")
plt.plot(d, w_mean_list_f3, 'k-',label ="mean squre curve of bayesian model of f3")

plt.xlim(0,12)
plt.ylim(5000,500000)
plt.legend()
plt.title("mse as a function of degree")
plt.ylabel("mean suqare error")
plt.xlabel("degree size")
plt.savefig('graph1.png')


plt.figure()
print("evidence for f3")
print(evi_list_3)
plt.plot(d, evi_list_3, 'r-',label ="the curve of log evidence of f3")

plt.xlim(0,12)
plt.ylim(-4000,-2000)
plt.legend()
plt.title("evidence as a function of degree size")
plt.ylabel("evidence value")
plt.xlabel("degree size ")
plt.savefig('graph2.png')




d = [1,2,3,4,5,6,7,8,9,10]
plt.figure()
print("")
print("mse of noregular model for f5")
print(w_noregular_list_f3)
print("mse of bayesian model for f5")
print(w_mean_list_f5)
#print(w_noregular_list_f3)
plt.plot(d, w_noregular_list_f5, 'b-',label ="mean squre curve of no regular model of f5")
plt.plot(d, w_mean_list_f5, 'k-',label ="mean squre curve of bayesian model of f5")

plt.xlim(0,12)
plt.ylim(5000,200000)
plt.legend()
plt.title("mse as a function of degree")
plt.ylabel("mean suqare error")
plt.xlabel("degree size")
plt.savefig('graph3.png')


plt.figure()
print("evidence for f5")
print(evi_list_5)
plt.plot(d, evi_list_5, 'r-',label ="the curve of log evidence of f5")

plt.xlim(0,12)
plt.ylim(-5000,-2000)
plt.legend()
plt.title("evidence as a function of degree size")
plt.ylabel("evidence value")
plt.xlabel("degree size ")
plt.savefig('graph4.png')
