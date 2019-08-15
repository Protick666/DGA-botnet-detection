from __future__ import print_function
import  enchant
import phrasefinder as pf
from math import pow

import phrasefinder as pf
import pickle
import gib_detect_train


badrepo={}




def vc_ck(string):
    num_vowels = 0
    for char in string:
        if char in "aeiouAEIOU":
            num_vowels = num_vowels + 1
    ans = num_vowels/(len(string) - num_vowels)
    return ans


def four_gram(temp):

    count=0
    for i in range(0, len(temp)-3):
        sub = temp[i:i+4]
        flag = 1
        for char in sub:
            if char in "aeiouAEIOU":
                flag = 2
                break
        if flag == 1:
            count = count + 1

    return (count/len(temp)) * 100

def count_bad(temp):
    global badrepo

    sum=0

    for j in range(0, len(temp)):
        for k in range(3, len(temp) - j + 1):
            sub = temp[j:j+k]

            if sub not in badrepo:
                break

            sum = sum + badrepo[sub] * 100 * (2**k)

    return sum/len(temp)


def mean(temp):
    global dct
    repo = []
    sum = 0
    for l in range(3, min(len(temp), 6)):
        for i in range(0, len(temp) - l):
            substr = temp[i:i + l]
            repo.append(substr)

    for word in repo:
        if dct.check(word) == True:
            sum += pow(2, len(word))
    sum = sum / len(temp)
    return sum

def freq_check(temp):

    repo = []
    sum = 0

    for l in range(3, min(len(temp), 5)):
        for i in range(0, len(temp) - l):
            substr = temp[i:i + l]
            repo.append(substr)

    for word in repo:
        query = word
        options = pf.SearchOptions()
        result = pf.search(pf.Corpus.AMERICAN_ENGLISH, query, options)
        maxno = -1
        for phrase in result.phrases:
            maxno = max(phrase.score, maxno)
        if (maxno < 0):
            continue
        sum += pow(2, len(word)) * maxno

    return sum

def check_gib(temp):
    global model_data
    query = temp

    model_mat = model_data['mat']
    threshold = model_data['thresh']
    sum = (gib_detect_train.avg_transition_prob(temp, model_mat))

    return sum



model_data = pickle.load(open('gib_model.pki', 'rb'))
dct = enchant.Dict("en_US")
outF = open("out.txt", "w")

count=0

with open('hello1.txt') as f:
    lines = f.readlines()
    for s in lines:
        j = 0
        for j in range(0,len(s)):
            if s[j] == '.':
                break
        temp = s[0:j]

        for j in range(0,len(temp)):
            for k in range(3,len(temp)-j+1):
                t=temp[j:j+k]
                if t in badrepo:
                    badrepo[t] = badrepo[t] + 1
                else:
                    badrepo[t] = 1


        '''
       
            s = s.substr(0, j);

            for (int j=0;j < s.length();j++)
            {
            for (unsigned int k=3;j+k <= s.length();k++)
            vis[s.substr(j, k)]++;

            }

        
        '''
        count = count + 1
        if count == 1000:
            break

count=0
with open('hello.txt') as f:
    lines = f.readlines()
    for s in lines:
        j = 0
        for j in range(0,len(s)):
            if s[j] == '.':
                break
        temp = s[0:j]

        l=len(temp)
        vc = vc_ck(temp)
        gm = four_gram(temp)
        mali_score = count_bad(temp)
        meaning_score = mean(temp)
        freq_score = freq_check(temp)
        gib_score = check_gib(temp)

        final_ans = str(l)+" "+str(vc)+" "+str(gm)+" "+str(mali_score)+" "+str(meaning_score)+" "+str(freq_score)
        final_ans = final_ans + " " + str(gib_score) + " 2"
        outF.write(final_ans)

        outF.write("\n")
        count=count+1
        print(count)
        if count == 20:
            break

outF.close()
