
import math
import pylab as plt
import numpy as np
# read data from training_data
fd = open("training_data.txt")

train_list = fd.read().strip().split()
# get lists for whole words


# read data from testing data

ft = open("test_data.txt")

test_list = ft.read().strip().split()
# get lists for whole words

test_n = len(test_list)
words_list = train_list + test_list


# set up dictionary map number to different words(10000)

diff_words = list(set(words_list))
d_n = len(diff_words)


reverse_index_words = {}

for index, word in enumerate(diff_words):
    reverse_index_words[word] = index


# set up different training set
n = len(train_list)
n_128 = n // 128
train_set_128 = words_list[: n_128]

# count mk for n/128 model
number_words_128 = {}

for word in train_set_128:
    if word not in number_words_128:
        number_words_128[word] = 0
    number_words_128[word] += 1


# build probability model for n/128 training set
p_pre_128_t_1, p_pre_128_t_2, p_pre_128_t_3, p_pre_128_t_4 = {},{},{},{}
p_pre_128_t_5, p_pre_128_t_6, p_pre_128_t_7, p_pre_128_t_8 = {},{},{},{}
p_pre_128_t_9, p_pre_128_t_10 = {},{}

pro = []
pro.append(0)
pro.append(p_pre_128_t_1)
pro.append(p_pre_128_t_2)
pro.append(p_pre_128_t_3)
pro.append(p_pre_128_t_4)
pro.append(p_pre_128_t_5)
pro.append(p_pre_128_t_6)
pro.append(p_pre_128_t_7)
pro.append(p_pre_128_t_8)
pro.append(p_pre_128_t_9)
pro.append(p_pre_128_t_10)

for a in range(1,11):
    for word in diff_words:
        index = reverse_index_words[word] 
        if word in number_words_128:
           pro[a][index] = (number_words_128[word] + a) / (n_128 + a * 10000)
        else:    
           pro[a][index] = (0 + a) / (n_128 + a * 10000)
         
pro_test = [0] * 11

#calculate logp(wi)
for a in range(1,11):
   for word in test_list:
      index = reverse_index_words[word]
      pro_test[a] += math.log(pro[a][index])
print(len(pro_test))
perplex_pre_128_test = [0] * 11

#calculate perplexity 

for a in range(1,11):   
   perplex_pre_128_test[a] = math.exp(- pro_test[a] / test_n)
new_perplex = perplex_pre_128_test[1:]



#evidence

evidence = [0] * 11
for a in range(1, 11):
    a_0 = a * d_n
    evidence[a] += math.log(math.factorial((a_0 - 1)))
    temp1 = 1

    for word in diff_words:
       if word in number_words_128:
          temp1 *= math.factorial((number_words_128[word] + a - 1))
       else:
          temp1 *= math.factorial(a - 1)
        
    
    evidence[a] += math.log(temp1)
    evidence[a] -= math.log(math.factorial((a_0 + n_128 -1)))
    temp2 = 1
    for word in diff_words:
       temp2 *= math.factorial((a - 1))

    evidence[a] -= math.log(temp2)

new_evidence = evidence[1:]


# draw figure
data = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]
plt.figure()
plt.plot(data, new_perplex , 'r-', label = "perplexity as a function of alpha")
plt.xlim(0,11)
new_ticks = np.linspace(0,11, 12)
plt.xticks(new_ticks)
plt.ylim((9700, 10110))
plt.legend()
plt.title("testing data")
plt.ylabel("perplexity")
plt.xlabel("alpha")
plt.savefig('graph3.png')


plt.figure()
plt.plot(data, new_evidence , 'r-', label = "evidence as a function of alpha")
plt.xlim(0,11)
new_ticks = np.linspace(0,11, 12)
plt.xticks(new_ticks)
plt.ylim((-46150,-45970))
plt.legend()
plt.title("training data")
plt.ylabel("log evidence")
plt.xlabel("alpha")
plt.savefig('graph4.png')
