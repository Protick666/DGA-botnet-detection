import  enchant
import phrasefinder as pf
from math import pow

d = enchant.Dict("en_US")



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
        for l in range (3,min(len(temp),6)):
            for i in range (0,len(temp)-l):
                substr=temp[i:i+l]
                repo.append(substr)

        for word in repo:
            if d.check(word) == True:
                sum += pow(2, len(word))
        sum=sum/len(temp)



        outF.write(str(sum))

        outF.write("\n")
        count=count+1
        print(count)

        if count == 1000:
            break

outF.close()








