# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 20:03:28 2017

@author: zheyiyi
"""



import math
# read data from training_data
fd = open("pg121.txt.clean",encoding = 'utf-8')

train_list = fd.read().strip().split()
n_121 = len(train_list)

# read data from test_data
fd1 = open("pg141.txt.clean",encoding = 'utf-8')
test1_list = fd1.read().strip().split()
n_141 = len(test1_list)

fd2 = open("pg1400.txt.clean",encoding = 'utf-8')
test2_list = fd2.read().strip().split()
n_1400 = len(test2_list)
total_list = test2_list + test1_list + train_list

diff_list = list(set(total_list))
s = len(diff_list)

index_words = {}
reverse_index_words = {}

for index, word in enumerate(diff_list):
    index_words[index] = word
    reverse_index_words[word] = index



train_dict = {}
for word in train_list:
    if word not in train_dict:
        train_dict[word] = 0
    train_dict[word] += 1

p_pre_121 = {}
for word in diff_list:
    index = reverse_index_words[word] 
    if word in train_dict:
        p_pre_121[index] = (train_dict[word] + 2) / (n_121 + 2 * s)
    else:
        p_pre_121[index] = 2 / (n_121 + 2 * s)
        
    

# perplexity for file pg141

p_test1 = 0

for word in test1_list:
   
      index = reverse_index_words[word] 
      p_test1 += math.log(p_pre_121[index])
   
   
perplexity_141 = math.exp(- p_test1 / n_141)
        



# perplexity for file pg1400

p_test2 = 0
for word in test2_list:
    
      index = reverse_index_words[word] 
      p_test2 += math.log(p_pre_121[index])

perplexity_1400 = math.exp(- p_test2 / n_1400)



if perplexity_1400 > perplexity_141:
    print("pg121 and pg_141 have the same author")
else:
    print("pg121 and pg_1400 have the same author")
    
    
"""

remove less than 50 times() 

"""

result = {}
remove_list = []

for word, number in train_dict.items():
    if number >= 50:      
        result[word] = number
    else:
        remove_list.append(word)
        
new_train_list = []

for word in train_list:
    if word not in remove_list:
        new_train_list.append(word)
        
        
p_pre_121_50 = {}
new_n_121 = len(new_train_list)


for word in diff_list:
    index = reverse_index_words[word] 
    if word in result:
        p_pre_121_50[index] = (result[word] + 2) / (new_n_121 + 2 * s)
    else:
        p_pre_121_50[index] =  2 / (new_n_121 + 2 * s)
   


# perplexity for file pg141


p_test1 = 0
for word in test1_list:

      index = reverse_index_words[word] 
      p_test1 += math.log(p_pre_121_50[index])

perplexity_141_50 = math.exp(- p_test1 / n_141)
        


# perplexity for file pg1400


p_test2 = 0
for word in test2_list:

      index = reverse_index_words[word] 
      p_test2 += math.log(p_pre_121_50[index])

perplexity_1400_50 = math.exp(- p_test2 / n_1400)



if perplexity_1400_50 > perplexity_141_50:
    print("pg121 and pg_141 have the same author")
else:
    print("pg121 and pg_1400 have the same author")



