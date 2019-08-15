
import pickle
import gib_detect_train

model_data = pickle.load(open('gib_model.pki', 'rb'))
'''
p=['hello',"gather","d1xx","ytjkacvzw"]
for word in p:
    model_mat = model_data['mat']
    threshold = model_data['thresh']
    print (gib_detect_train.avg_transition_prob(word, model_mat) )
'''
outF = open("gibberish.txt", "w")


count=0

with open('hello.txt') as f:
    lines = f.readlines()
    for s in lines:
        j = 0
        for j in range(0, len(s)):
            if s[j] == '.':
                break
        temp = s[0:j]




        query = temp


        model_mat = model_data['mat']
        threshold = model_data['thresh']
        sum=(gib_detect_train.avg_transition_prob(temp, model_mat))

        outF.write(str(sum))

        outF.write("\n")
        count=count+1
        print(count)

        if count == 1000:
            break

outF.close()





