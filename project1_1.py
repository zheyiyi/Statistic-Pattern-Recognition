# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 11:36:59 2017

@author: zheyiyi
"""


import math
import pylab as plt

# read data from training_data
fd = open("training_data.txt")
train_list = fd.read().strip().split()

# read data from testing data

ft = open("test_data.txt")
test_list = ft.read().strip().split()

n_t = len(test_list)
words_list = train_list + test_list


# set up dictionary map number to different words(10000)

diff_words = list(set(words_list))

index_words = {}
reverse_index_words = {}

for index, word in enumerate(diff_words):
    index_words[index] = word
    reverse_index_words[word] = index



# set up different training set
n = len(train_list)

n_128 = n // 128
n_64 = n // 64
n_16 = n // 16
n_4 = n // 4
n_1 = n


train_set_128 = words_list[: n_128]
train_set_64 = words_list[: n_64]
train_set_16 = words_list[: n_16]
train_set_4 = words_list[: n_4]
train_set_1 = words_list[: n]


perplex_ml_data = []
perplex_map_data = []
perplex_pre_data = []

# count mk for n/128 model
number_words_128 = {}
for word in train_set_128:
    if word not in number_words_128:
        number_words_128[word] = 0
    number_words_128[word] += 1


        
p_ml_128_t = {}
p_map_128_t = {}
p_pre_128_t = {}


# build probability model for n/128 training set
for word in diff_words:
    index = reverse_index_words[word] 
    if word in number_words_128:
        p_ml_128_t[index] = number_words_128[word] / n_128
        p_map_128_t[index] = (number_words_128[word] + 2 - 1) / (n_128 + 2 * 10000 - 10000)
        p_pre_128_t[index] = (number_words_128[word] + 2) / (n_128 + 2 * 10000)
    else:    
        p_ml_128_t[index] = 0
        p_map_128_t[index] = (0 + 2 - 1) / (n_128 + 2 * 10000 - 10000)
        p_pre_128_t[index] = (0 + 2) / (n_128 + 2 * 10000)



   
perplex_ml_128 = 0
p_ml_128 = 0
perplex_map_128 = 0
p_map_128 = 0
perplex_pre_128 = 0
p_pre_128 = 0


#calculate logp(wi)
for word in train_set_128:
    index = reverse_index_words[word]
    p_ml_128 += math.log(p_ml_128_t[index])    
    p_map_128 += math.log(p_map_128_t[index])
    p_pre_128 += math.log(p_pre_128_t[index])

#calculate perplexity     

perplex_ml_128 = math.exp(- p_ml_128 / n_128)
perplex_map_128 = math.exp(- p_map_128 / n_128)
perplex_pre_128 = math.exp(- p_pre_128 / n_128)

perplex_ml_data.append(perplex_ml_128)
perplex_map_data.append(perplex_map_128)
perplex_pre_data.append(perplex_pre_128)




# for set n/64

number_words_64 = {}
for word in train_set_64:
    if word not in number_words_64:
        number_words_64[word] = 0
    number_words_64[word] += 1


p_ml_64_t = {}
p_map_64_t = {}
p_pre_64_t = {}
for word in diff_words:
    index = reverse_index_words[word] 
    if word in number_words_64:
        p_ml_64_t[index] = number_words_64[word] / n_64
        p_map_64_t[index] = (number_words_64[word] + 2 - 1) / (n_64 + 2 * 10000 - 10000)
        p_pre_64_t[index] = (number_words_64[word] + 2) / (n_64 + 2 * 10000)
    else:
        
        p_ml_64_t[index] = 0
        p_map_64_t[index] = (0 + 2 - 1) / (n_64 + 2 * 10000 - 10000)
        p_pre_64_t[index] = (0 + 2) / (n_64 + 2 * 10000)
    

perplex_ml_64 = 0
p_ml_64 = 0
perplex_map_64 = 0
p_map_64 = 0
perplex_pre_64 = 0
p_pre_64 = 0


for word in train_set_64:
    index = reverse_index_words[word]
    p_ml_64 += math.log(p_ml_64_t[index])    
    p_map_64 += math.log(p_map_64_t[index])
    p_pre_64 += math.log(p_pre_64_t[index])
    
perplex_ml_64 = math.exp(- p_ml_64 / n_64)
perplex_map_64 = math.exp(- p_map_64 / n_64)
perplex_pre_64 = math.exp(- p_pre_64 / n_64)

perplex_ml_data.append(perplex_ml_64)
perplex_map_data.append(perplex_map_64)
perplex_pre_data.append(perplex_pre_64)
print(perplex_ml_64)
print(perplex_pre_64)
print(perplex_map_64)

number_words_16 = {}
for word in train_set_16:
    if word not in number_words_16:
        number_words_16[word] = 0
    number_words_16[word] += 1


p_ml_16_t = {}
p_map_16_t = {}
p_pre_16_t = {}
for word in diff_words:
    index = reverse_index_words[word]
    if word in number_words_16:
       p_ml_16_t[index] = number_words_16[word] / n_16      
       p_map_16_t[index] = (number_words_16[word] + 2 - 1) / (n_16 + 2 * 10000 - 10000)
       p_pre_16_t[index] = (number_words_16[word] + 2) / (n_16 + 2 * 10000)
    
    else:
       p_ml_16_t[index] = 0 / n_16  
       p_map_16_t[index] = (0 + 2 - 1) / (n_16 + 2 * 10000 - 10000)
       p_pre_16_t[index] = (0 + 2) / (n_16 + 2 * 10000)
    

perplex_ml_16 = 0
p_ml_16 = 0
perplex_map_16 = 0
p_map_16 = 0
perplex_pre_16 = 0
p_pre_16 = 0


for word in train_set_16:
    index = reverse_index_words[word]
    p_ml_16 += math.log(p_ml_16_t[index])    
    p_map_16 += math.log(p_map_16_t[index])
    p_pre_16 += math.log(p_pre_16_t[index])

perplex_ml_16 = math.exp(- p_ml_16 / n_16)
perplex_map_16 = math.exp(- p_map_16 / n_16)
perplex_pre_16 = math.exp(- p_pre_16 / n_16)

perplex_ml_data.append(perplex_ml_16)
perplex_map_data.append(perplex_map_16)
perplex_pre_data.append(perplex_pre_16)



# for set n/4

number_words_4 = {}
for word in train_set_4:
    if word not in number_words_4:
        number_words_4[word] = 0
    number_words_4[word] += 1


p_ml_4_t = {}
p_map_4_t = {}
p_pre_4_t = {}
for word in diff_words:
   index = reverse_index_words[word] 
   if word in number_words_4:
     
      p_ml_4_t[index] = number_words_4[word] / n_4
      p_map_4_t[index] = (number_words_4[word] + 2 - 1) / (n_4 + 2 * 10000 - 10000)
      p_pre_4_t[index] = (number_words_4[word] + 2) / (n_4 + 2 * 10000)
   else:
      p_ml_4_t[index] = 0
      p_map_4_t[index] = (0 + 2 - 1) / (n_4 + 2 * 10000 - 10000)
      p_pre_4_t[index] = (0 + 2) / (n_4 + 2 * 10000)
      
    

perplex_ml_4 = 0
p_ml_4 = 0
perplex_map_4 = 0
p_map_4 = 0
perplex_pre_4 = 0
p_pre_4 = 0


for word in train_set_4:
    index = reverse_index_words[word]
    p_ml_4 += math.log(p_ml_4_t[index])    
    p_map_4 += math.log(p_map_4_t[index])
    p_pre_4 += math.log(p_pre_4_t[index])
    
perplex_ml_4 = math.exp(- p_ml_4 / n_4)
perplex_map_4 = math.exp(- p_map_4 / n_4)
perplex_pre_4 = math.exp(- p_pre_4 / n_4)

perplex_ml_data.append(perplex_ml_4)
perplex_map_data.append(perplex_map_4)
perplex_pre_data.append(perplex_pre_4)

# for set n


number_words_1 = {}
for word in train_set_1:
    if word not in number_words_1:
        number_words_1[word] = 0
    number_words_1[word] += 1

p_ml_1_t = {}
p_map_1_t = {}
p_pre_1_t = {}
for word in diff_words:
    index = reverse_index_words[word] 
    if word in number_words_1:
       p_ml_1_t[index] = number_words_1[word] / n_1    
       p_map_1_t[index] = (number_words_1[word] + 2 - 1) / (n_1 + 2 * 10000 - 10000)
       p_pre_1_t[index] = (number_words_1[word] + 2) / (n_1 + 2 * 10000)
    else:
       p_ml_1_t[index] = 0   
       p_map_1_t[index] = (0 + 2 - 1) / (n_1 + 2 * 10000 - 10000)
       p_pre_1_t[index] = (0 + 2) / (n_1 + 2 * 10000)
    

perplex_ml_1 = 0
p_ml_1 = 0
perplex_map_1 = 0
p_map_1 = 0
perplex_pre_1 = 0
p_pre_1 = 0



for word in train_set_1:
    index = reverse_index_words[word]
    p_ml_1 += math.log(p_ml_1_t[index])    
    p_map_1 += math.log(p_map_1_t[index])
    p_pre_1 += math.log(p_pre_1_t[index])


perplex_ml_1 = math.exp(- p_ml_1 / n_1)
perplex_map_1 = math.exp(- p_map_1 / n_1)
perplex_pre_1 = math.exp(- p_pre_1 / n_1)


perplex_ml_data.append(perplex_ml_1)
perplex_map_data.append(perplex_map_1)
perplex_pre_data.append(perplex_pre_1)

# draw figure

data = [n_128,n_64, n_16, n_4, n]

plt.figure()
plt.plot(data, perplex_ml_data, 'r-', label = "mle")
plt.plot(data, perplex_map_data, 'b-', label = "map")
plt.plot(data, perplex_pre_data, 'k-', label = "pred.")

plt.xlim(0,650000)
plt.ylim((3000,9000))
plt.legend(loc = 'lower right')
plt.title("training data")
plt.ylabel("perplexity")
plt.xlabel("size of trainig data")
plt.savefig('graph1.png')



"""
testing data
"""


p_test_ml_128 = 0
p_test_ml_64 = 0
p_test_ml_16 = 0
p_test_ml_4 = 0
p_test_ml_1 = 0

p_test_map_128 = 0
p_test_map_64 = 0
p_test_map_16 = 0
p_test_map_4 = 0
p_test_map_1 = 0

p_test_pre_128 = 0
p_test_pre_64 = 0
p_test_pre_16 = 0
p_test_pre_4 = 0
p_test_pre_1 = 0

#calculate the log p(w1) for every word in testing data
for word in test_list:
    index = reverse_index_words[word] 
    if p_ml_128_t[index] == 0:
       p_test_ml_128 += - math.inf
    else:
       p_test_ml_128 += math.log(p_ml_128_t[index])
    p_test_map_128 += math.log(p_map_128_t[index])
    p_test_pre_128 += math.log(p_pre_128_t[index])
        
    if p_ml_64_t[index] == 0:
       p_test_ml_64 += - math.inf
    else:
       p_test_ml_64 += math.log(p_ml_64_t[index])
    p_test_map_64 += math.log(p_map_64_t[index])
    p_test_pre_64 += math.log(p_pre_64_t[index])
    
    if p_ml_16_t[index] == 0:
       p_test_ml_16 += - math.inf
    else:
       p_test_ml_16 += math.log(p_ml_16_t[index])
    p_test_map_16 += math.log(p_map_16_t[index])
    p_test_pre_16 += math.log(p_pre_16_t[index])
    
    if p_ml_4_t[index] == 0:
       p_test_ml_4 += - math.inf
    else:
       p_test_ml_4 += math.log(p_ml_4_t[index])
    p_test_map_4 += math.log(p_map_4_t[index])
    p_test_pre_4 += math.log(p_pre_4_t[index])
    
    if p_ml_1_t[index] == 0:
       p_test_ml_1 += - math.inf
    else:
       p_test_ml_1 += math.log(p_ml_1_t[index])
    p_test_map_1 += math.log(p_map_1_t[index])
    p_test_pre_1 += math.log(p_pre_1_t[index])  



perplex_ml_128_test = math.exp(- p_test_ml_128 / n_t)
perplex_map_128_test = math.exp(- p_test_map_128 / n_t)
perplex_pre_128_test = math.exp(- p_test_pre_128 / n_t)

perplex_ml_64_test = math.exp(- p_test_ml_64 / n_t)
perplex_map_64_test = math.exp(- p_test_map_64 / n_t)
perplex_pre_64_test = math.exp(- p_test_pre_64 / n_t)

perplex_ml_16_test = math.exp(- p_test_ml_16 / n_t)
perplex_map_16_test = math.exp(- p_test_map_16 / n_t)
perplex_pre_16_test = math.exp(- p_test_pre_16 / n_t)

perplex_ml_4_test = math.exp(- p_test_ml_4 / n_t)
perplex_map_4_test = math.exp(- p_test_map_4 / n_t)
perplex_pre_4_test = math.exp(- p_test_pre_4 / n_t)


perplex_ml_1_test = math.exp(- p_test_ml_1 / n_t)
perplex_map_1_test = math.exp(- p_test_map_1 / n_t)
perplex_pre_1_test = math.exp(- p_test_pre_1 / n_t)


perplex_ml_test = []
perplex_ml_test.append(perplex_ml_128_test)
perplex_ml_test.append(perplex_ml_64_test)
perplex_ml_test.append(perplex_ml_16_test)
perplex_ml_test.append(perplex_ml_4_test)
perplex_ml_test.append(perplex_ml_1_test)

perplex_map_test = []
perplex_map_test.append(perplex_map_128_test)
perplex_map_test.append(perplex_map_64_test)
perplex_map_test.append(perplex_map_16_test)
perplex_map_test.append(perplex_map_4_test)
perplex_map_test.append(perplex_map_1_test)

perplex_pre_test = []
perplex_pre_test.append(perplex_pre_128_test)
perplex_pre_test.append(perplex_pre_64_test)
perplex_pre_test.append(perplex_pre_16_test)
perplex_pre_test.append(perplex_pre_4_test)
perplex_pre_test.append(perplex_pre_1_test)


plt.figure()
plt.plot(data, perplex_ml_test, 'r-', label = "mle")
plt.plot(data, perplex_map_test, 'b-', label = "map")
plt.plot(data, perplex_pre_test, 'k-', label = "pred.")

plt.xlim(0,650000)
plt.ylim((8500,10500))
plt.legend()
plt.title("testing data")
plt.ylabel("perplexity")
plt.xlabel("size of trainig data")
plt.savefig('graph2.png')
