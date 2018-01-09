# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 00:12:33 2017

@author: zheyiyi
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from finalproject import train
from svd import svd_train
result1_100k = []
#result1_1M = []
result2_100k = []
#result2_1M = []
rank = []

for i in range(1,10): 
    min_value, u, v, bu, bi = svd_train( "u.data", i) 
    new_predict, u, v = train("u.data", i)
    result1_100k.append(min_value)
    result2_100k.append(new_predict)
    
    #min_value, u, v, bu, bi = svd_train("ml-1m", "ratings.dat", i) 
    #new_predict, u, v = train("ml-1m", "ratings.dat", i)
    #result1_1M.append(min_value)
    #result2_1M.append(new_predict)   
    rank.append(i)
    #print(i)


plt.figure()

print(result1_100k,"svd")
print(result2_100k, "variationial")
plt.plot(rank, result1_100k, 'r-',label ="the rmse of svd as a function of rank")
plt.plot(rank, result2_100k, 'k-',label ="the rmse of variance as a function of rank")

plt.xlim(1,11)
plt.ylim(0.9,1.1)
plt.legend()
plt.title("Figure 1")
plt.ylabel("rmse")
plt.xlabel("rank size")
plt.savefig('graph1.png')

"""
fig = plt.figure()

print(result1_100k,"svd")
print(result2_100k,"variance")
print(result1_1M,"svd")
print(result2_1M,"variance")


ax1 = fig.add_subplot(221)
ax1.plot(rank, result1_100k, 'r-',label ="variance")
ax1.plot(rank, result2_100k, 'k-',label ="svd")
plt.xlim(1,6)
plt.ylim(0.9,1.1)
plt.legend()
plt.title("move_1k")
plt.ylabel("rmse")
plt.xlabel("rank size")
plt.savefig('graph1.png')

ax2 = fig.add_subplot(222)
ax2.plot(rank, result1_1M, 'r-',label ="variance")
ax2.plot(rank, result2_1M, 'k-',label ="svd")
plt.xlim(1,6)
plt.ylim(0.9,1.1)
plt.legend()
plt.title("move_1m")
plt.ylabel("rmse")
plt.xlabel("rank size")
plt.savefig('graph2.png')
plt.tight_layout()
fig = plt.gcf()
"""