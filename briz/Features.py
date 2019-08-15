from __future__ import print_function
import  enchant
from math import pow
import pickle
import phrasefinder as pf
import gib_detect_train




def vowel_consonant_ratio(temp):
    num_vowels = 0
    for char in temp:
        if char in "aeiouAEIOU":
            num_vowels = num_vowels + 1
    if len(temp) - num_vowels == 0:
        return 1
    ans = num_vowels/(len(temp) - num_vowels)
    return ans


def four_gram(temp):
    cnt = 0
    for i in range(0, len(temp)-3):
        sub = temp[i:i+4]
        flag = 1
        for char in sub:
            if char in "aeiouAEIOU":
                flag = 2
                break
        if flag == 1:
            cnt = cnt + 1

    return (cnt/len(temp)) * 100


def malicious_word_presence(temp):
    global bad_repo

    sum = 0
    for j in range(0, len(temp)):
        for k in range(3, len(temp) - j + 1):
            sub = temp[j:j+k]
            if sub not in bad_repo:
                break
            sum = sum + bad_repo[sub] * 100 * (2**k)

    return sum / len(temp)


def meaning(temp):
    global dct
    repo = []
    sum = 0
    for l in range(3, min(len(temp), 6)):
        for i in range(0, len(temp) - l):
            substr = temp[i:i + l]
            repo.append(substr)

    for word in repo:
        if dct.check(word):
            sum += pow(2, len(word))
    sum = sum / len(temp)
    return sum


def google_corpus_freq(temp):
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
        max_no = -1
        for phrase in result.phrases:
            max_no = max(phrase.score, max_no)
        if max_no < 0:
            continue
        sum += pow(2, len(word)) * max_no

    return sum


def markov_model(temp):
    global model_data
    # query = temp
    model_mat = model_data['mat']
    # threshold = model_data['thresh']
    sum = (gib_detect_train.avg_transition_prob(temp, model_mat))

    return sum


off = ["conflicker_features.txt", "500KL3_features.txt", "kwyjibo_features.txt"];
iff = ["conflicker.txt", "500KL3.txt", "kwyjibo.txt"];


for it in range(0,4):
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

                try:
                    l = len(string)
                    vc = vowel_consonant_ratio(string)
                    gm = four_gram(string)
                    # mali_score = malicious_word_presence(string)
                    meaning_score = meaning(string)
                    freq_score = google_corpus_freq(string)
                    gib_score = markov_model(string)

                except Exception as e:
                    l = 0
                    vc = 0
                    gm = 0
                    meaning_score = 0
                    freq_score = 0
                    mali_score = 0
                    gib_score = 0

                final_ans = str(l)+" "+str(vc)+" "+str(gm)+" "+str(meaning_score)+" "+str(freq_score)
                final_ans = final_ans + " " + str(gib_score) + " 1"  # change class [0 for benign, 1 for mal]
                outF.write(final_ans)

                outF.write("\n")
                print(count, s)
                count = count+1
                if count == 2000:  # change line no
                    break

        outF.close()
