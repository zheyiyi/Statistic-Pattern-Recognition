# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 07:01:45 2017

@author: zheyiyi
"""
#part 1.

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
        return np.array(list_data, dtype=float)

def mean_squre(x,y,w):
    sum = 0
    n = len(y)
    for i in range(n):
        a = np.dot(x[i].T,w)
        sum += math.pow((a - y[i]),2)
    return sum / n


#lambda from 0 to 150

def regress_regular(x,y,z):
    result = []
    for i in range(z):
        t = np.dot(x.T,x)
        a = i * np.identity(len(t[0])) + t
        inverse_value = np.linalg.inv(a)
        w = np.dot(np.dot(inverse_value,x.T),y)
        result.append(w)
    return result
      
# training set mse as a function of the regularization parameter
# train_100_10 and trainr_100_10 and test

train_100_10 = open_matrix("train-100-10.csv")
trainr_100_10 = open_matrix("trainR-100-10.csv")
w_total = regress_regular(train_100_10, trainr_100_10, 151)

result_100_10 = []
for w in w_total:
    b = mean_squre(train_100_10, trainr_100_10, w)
    result_100_10.append(b)
    
resultr_100_10 = []
test_100_10 = open_matrix("test-100-10.csv")
testr_100_10 = open_matrix("testR-100-10.csv")

for w in w_total:
    b = mean_squre(test_100_10, testr_100_10, w)
    resultr_100_10.append(b)

# train_100_100 and trainr_100_100 and test

train_100_100= open_matrix("train-100-100.csv")
trainr_100_100 = open_matrix("trainR-100-100.csv")
w_total = regress_regular(train_100_100, trainr_100_100, 151)

result_100_100 = []
for w in w_total:
    b = mean_squre(train_100_100, trainr_100_100, w)
    result_100_100.append(b)
    
resultr_100_100 = []
test_100_100 = open_matrix("test-100-100.csv")
testr_100_100 = open_matrix("testR-100-100.csv")

for w in w_total:
    b = mean_squre(test_100_100, testr_100_100, w)
    resultr_100_100.append(b)

        
# train_1000_100 and trainr_1000_100 and test

train_1000_100= open_matrix("train-1000-100.csv")
trainr_1000_100 = open_matrix("trainR-1000-100.csv")
w_total = regress_regular(train_1000_100, trainr_1000_100, 151)

result_1000_100 = []
for w in w_total:
    b = mean_squre(train_1000_100, trainr_1000_100, w)
    result_1000_100.append(b)
    
resultr_1000_100 = []
test_1000_100 = open_matrix("test-1000-100.csv")
testr_1000_100 = open_matrix("testR-1000-100.csv")

for w in w_total:
    b = mean_squre(test_1000_100, testr_1000_100, w)
    resultr_1000_100.append(b)

# train-crime.csv and test-crime.csv and and test

train_crime= open_matrix("train-crime.csv")
trainr_crime = open_matrix("trainR-crime.csv")
w_total = regress_regular(train_crime, trainr_crime, 151)

result_crime = []
for w in w_total:
    b = mean_squre(train_crime, trainr_crime, w)
    result_crime.append(b)
    
resultr_crime = []
test_crime = open_matrix("test-crime.csv")
testr_crime = open_matrix("testR-crime.csv")

for w in w_total:
    b = mean_squre(test_crime, testr_crime, w)
    resultr_crime.append(b)

# train-wine.csv and test-wine.csv and and test

train_wine= open_matrix("train-wine.csv")
trainr_wine = open_matrix("trainR-wine.csv")
w_total = regress_regular(train_wine, trainr_wine, 151)

result_wine = []
for w in w_total:
    b = mean_squre(train_wine, trainr_wine, w)
    result_wine.append(b)
    
resultr_wine = []
test_wine = open_matrix("test-wine.csv")
testr_wine = open_matrix("testR-wine.csv")

for w in w_total:
    b = mean_squre(test_wine, testr_wine, w)
    resultr_wine.append(b)

lambda_150 = []
for i in range(151):
    lambda_150.append(i)


plt.figure()
print("mse of train_100_10")
print(result_100_10)
print("mse of test_100_10")
print(resultr_100_10)
plt.plot(lambda_150, result_100_10, 'b-',label ="train-100-10")
plt.plot(lambda_150, resultr_100_10, 'r-',label ="test-100-10")
plt.xlim(0,154)
plt.ylim(0,6)
plt.legend()
plt.title("mse as a function of lambda")
plt.ylabel("mean suqare error")
plt.xlabel("the value of lambda")
plt.savefig('graph1.png')

plt.figure()
print("mse of train_100_100")
print(result_100_100)
print("mse of test_100_100")
print(resultr_100_100)
plt.plot(lambda_150, result_100_100, 'b-',label ="train-100-100")
plt.plot(lambda_150, resultr_100_100, 'r-',label ="test-100-100")
plt.xlim(0,154)
plt.ylim(0,10)
plt.legend()
plt.title("mse as a function of lambda")
plt.ylabel("mean suqare error")
plt.xlabel("the value of lambda")
plt.savefig('graph1.png')


plt.figure()
print("mse of train_1000_100")
print(result_1000_100)
print("mse of test_1000_100")
print(resultr_1000_100)
plt.plot(lambda_150, result_1000_100, 'b-',label ="train-1000-100")
plt.plot(lambda_150, resultr_1000_100, 'r-',label ="test-1000-100")
plt.xlim(0,154)
plt.ylim(0,6)
plt.legend()
plt.title("mse as a function of lambda")
plt.ylabel("mean suqare error")
plt.xlabel("the value of lambda")
plt.savefig('graph1.png')

plt.figure()

print("mse of train_crime")
print(result_crime)
print("mse of test_crime")
print(resultr_crime)
plt.plot(lambda_150, result_crime, 'b-',label ="train-crime")
plt.plot(lambda_150, resultr_crime, 'r-',label ="test-crime")
plt.xlim(0,154)
plt.ylim(0,0.6)
plt.legend()
plt.title("mse as a function of lambda")
plt.ylabel("mean suqare error")
plt.xlabel("the value of lambda")
plt.savefig('graph1.png')

plt.figure()
print("mse of train_wine")
print(result_crime)
print("mse of test_wine")
print(resultr_wine)
plt.plot(lambda_150, result_wine, 'b-',label ="train-wine")
plt.plot(lambda_150, resultr_wine, 'r-',label ="test-wine")
plt.xlim(0,154)
plt.ylim(0.5,0.7)
plt.legend()
plt.title("mse as a function of lambda")
plt.ylabel("mean suqare error")
plt.xlabel("the value of lambda")
plt.savefig('graph2.png')

#print(resultr_1000_100)


#part 2
lambda_small = 10
lambda_right = np.argmin(resultr_1000_100)
lambda_big = 100

#print(lambda_big)
lambda_right_100_10 = np.argmin(resultr_100_10)
lambda_right_100_100 = np.argmin(resultr_100_100)

lambda_right_crime = np.argmin(resultr_crime)
lambda_right_wine = np.argmin(resultr_wine)


print(resultr_100_10[lambda_right_100_10], lambda_right_100_10)
print(resultr_100_100[lambda_right_100_100], lambda_right_100_100)
print(resultr_1000_100[lambda_right], lambda_right)
print(resultr_crime[lambda_right_crime], lambda_right_crime)
print(resultr_wine[lambda_right_wine], lambda_right_wine)

train_size = []
for i in range(10, 801, 10):
    train_size.append(i)
c = range(0, 1000)

def matrixize(k):
    train_list = []
    tlabel_list = []
    numbers = random.sample(c,k)
  
    for num in numbers:
        
        train_list.append(train_1000_100[num])
        
        tlabel_list.append(trainr_1000_100[num])
    train_np = np.array(train_list)
    tlabel_np = np.array(tlabel_list)
    return (train_np, tlabel_np)

def re_regression(x,y,z):
    t = np.dot(x.T,x)
    a = np.add(z * np.identity(len(t[0])), t)
    inverse_value = np.linalg.inv(a)
    w = np.dot(np.dot(inverse_value,x.T),y)
    return mean_squre(test_1000_100, testr_1000_100,w)

    
#k is the training size we choose and z is the value of lambda
def repeat(z,k):
    sum = 0 
    for i in range(10):
        data, label = matrixize(k)
        mse = re_regression(data, label, z)
        sum += mse
    return sum / 10

def get_mse(z):
    mse_set = []
    for size in train_size:
        mse = repeat(z, size)
        mse_set.append(mse)
    return mse_set

mse_1000_100_small = get_mse(lambda_small)
mse_1000_100_right = get_mse(lambda_right)
mse_1000_100_big = get_mse(lambda_big)

#print(mse_1000_100_right[27])
print("mse of test_1000_100 for lambda = 10")
print(mse_1000_100_small)
print("mse of test_1000_100 for lambda = 27")
print(mse_1000_100_right)
print("mse of test_1000_100 for lambda = 100")
print(mse_1000_100_big)

plt.figure()
plt.plot(train_size, mse_1000_100_small, 'r-',label ="lambda = 10")
plt.plot(train_size, mse_1000_100_right, 'b-',label ="lambda = 27")
plt.plot(train_size, mse_1000_100_big, 'k-',label ="lambda = 100")

plt.xlim(0,800)
plt.ylim(4,9)
plt.legend()
plt.title("mse as a function of training_size")
plt.ylabel("mean suqare error")
plt.xlabel("training size")
plt.savefig('graph3.png')
    
lambda_right_10 = np.min(resultr_100_10)
lambda_right_100 = np.min(resultr_100_100)
lambda_right_crime =  np.min(resultr_crime)
lambda_right_wine =  np.min(resultr_wine)

    
        
