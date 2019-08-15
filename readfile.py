from __future__ import print_function
import phrasefinder as pf
from math import pow
outF = open("out.txt", "w")


count=0

with open('hello.txt') as f:
    lines = f.readlines()
    for s in lines:
        j = 0
        for j in range(0,len(s)):
            if s[j] == '.':
                break
        temp = s[0:j]
        repo=[]
        sum=0
        for l in range (3,min(len(temp),5)):
            for i in range (0,len(temp)-l):
                substr=temp[i:i+l]
                repo.append(substr)
        for word in repo:
            query = word
            options = pf.SearchOptions()
            result = pf.search(pf.Corpus.AMERICAN_ENGLISH, query, options)
            maxno=-1
            for phrase in result.phrases:
                maxno = max(phrase.score,maxno)
            if(maxno < 0):
                continue
            sum+= pow(10,len(word))*maxno

        outF.write(str(sum))

        outF.write("\n")
        count=count+1
        print(count)

        if count == 1000:
            break

outF.close()








