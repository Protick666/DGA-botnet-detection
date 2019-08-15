from __future__ import print_function
import  enchant

import  enchant
from math import pow
import pickle
import phrasefinder as pf
import gib_detect_train

def ck(temp):
    global dct
    global D
    repo = []
    sum = 0
    for l in range(3, min(len(temp), 6)):
        for i in range(0, len(temp) - l):
            substr = temp[i:i + l]
            repo.append(substr)

    repo = list(set(repo))

    for w1 in repo:
        for w2 in repo:
            if w1 == w2:
                continue
            if w1 not in D:
                continue
            if w2 not in D[w1]:
                continue
            sum += len(w1) * len(w2) * D[w1][w2]

    return sum



with open("big.txt") as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line

temp = []

cnt = 0
for w in content:
    cnt = cnt+1
    #print(w,cnt)
    if cnt == 3000:
        break
    p = w.split()
    for t in p:
        temp.append(t)

Word = []
t = []
pun = ",?:."

for p in temp:
    print(p)
    if p[len(p) - 1] in pun:
        t.append(p[0:len(p)-1])
        g = t.copy();
        Word.append(g)
        t.clear()
    else:
        t.append(p)

D = {}

for line in Word:
    for w in line:
        for x in line:
            if x == w:
                continue
            #print(x,w)
            if w not in D:
                D[w] = {}
            if x not in D[w]:
                D[w][x] = 0
            D[w][x] = D[w][x] + 1


dct = enchant.Dict("en_US")






off = ["pcfg_dict_correlation_scores.txt", "pcfg_dict_num_correlation_scores.txt", "pcfg_ipv4_correlation_scores.txt" , "pcfg_ipv4_num_correlation_scores.txt", "srizbi_correlation_scores.txt", "torpig_correlation_scores.txt", "zeus_correlation_scores.txt", "kraken_correlation_scores.txt", "DNL1_correlation_scores.txt", "DNL2_correlation_scores.txt", "DNL3_correlation_scores.txt", "DNL4_correlation_scores.txt", "500KL1_correlation_scores.txt", "500KL2_correlation_scores.txt", "500KL3_correlation_scores.txt", "9ML1_correlation_scores.txt"];
iff = ["pcfg_dict.txt", "pcfg_dict_num.txt", "pcfg_ipv4.txt" , "pcfg_ipv4_num.txt", "srizbi.txt", "torpig.txt", "zeus.txt", "kraken.txt", "DNL1.txt", "DNL2.txt", "DNL3.txt", "DNL4.txt", "500KL1.txt", "500KL2.txt", "500KL3.txt", "9ML1.txt"];


for it in range(0,16):
        bad_repo = {}
        model_data = pickle.load(open('gib_model.pki', 'rb'))
        dct = enchant.Dict("en_US")
        outF = open(off[it],"w")


        with open(iff[it]) as f:  # change file name
            lines = f.readlines()
            count = 0
            for s in lines:
                j = 0
                for j in range(0,len(s)):
                    if s[j] == '.':
                        break

                string = s[0:j]

                sc = ck(string)
                #print(sc, string)

                final_ans = str(sc) + " 1"  # change class [0 for benign, 1 for mal]
                outF.write(final_ans)

                outF.write("\n")
                #print(count, s)
                count = count+1
                if count == 2000:  # change line no
                    break

        outF.close()







