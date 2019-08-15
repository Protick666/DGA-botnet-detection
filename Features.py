from __future__ import print_function
import enchant
from math import pow
import pickle
import gzip
import gensim
import logging

import phrasefinder as pf
import gib_detect_train

bad_repo = {}
model_data = pickle.load(open('gib_model.pki', 'rb'))
dct = enchant.Dict("en_US")
DICTIONARY = "dicti.txt"
# Keep some interesting statistics
NodeCount = 0
WordCount = 0
# The Trie data structure keeps a set of words, organized with one node for
# each letter. Each node has a branch for each letter that may follow it in the
# set of words.


class TrieNode:
    def __init__(self):
        self.word = None
        self.children = {}

        global NodeCount
        NodeCount += 1

    def insert(self, word):
        node = self
        for letter in word:
            if letter not in node.children:
                node.children[letter] = TrieNode()

            node = node.children[letter]

        node.word = word

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)


def show_file_contents(input_file):
    with gzip.open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            print(line)
            break


def read_input(input_file):
    """This method reads the input file which is in gzip format"""
    logging.info("reading file {0}...this may take a while".format(input_file))
    with gzip.open(input_file, 'rb') as f:
        for i, line in enumerate(f):

            if (i % 10000 == 0):
                logging.info("read {0} reviews".format(i))
            #if i == 10000:
             #   break
            # do some pre-processing and return list of words for each review
            # text
            yield gensim.utils.simple_preprocess(line)


data_file = "reviews_data.txt.gz"
# read the tokenized reviews into a list
# each review item becomes a serries of words
# so this becomes a list of lists
documents = list(read_input(data_file))
logging.info("Done reading data file")
model = gensim.models.Word2Vec(
        documents,
        size=5,
        window=10,
        min_count=2,
        workers=4)


def vowel_consonant_ratio(temp):
    num_vowels = 0
    for char in temp:
        if char in "aeiouAEIOU":
            num_vowels = num_vowels + 1
    if len(temp) - num_vowels == 0:
        return 1
    ans = num_vowels/(len(temp) - num_vowels)
    return ans


def four_gram_score(temp):
    cnt = 0
    for i in range(0, len(temp)-3):
        sub = temp[i: i+4]
        # print(sub)
        flag = 1
        for char in sub:
            if char in "aeiouAEIOU":
                flag = 2
                break
        if flag == 1:
            cnt = cnt + 1

    return (cnt/len(temp)) * 100


def meaning_score(temp):
    global dct
    repo = [temp]
    sum = 0
    for l in range(3, len(temp)):
        for i in range(0, len(temp)-l+1):
            substr = temp[i:i + l]
            # print(substr)
            repo.append(substr)

    for word in repo:
        if dct.check(word):
            sum += pow(2, len(word))
    sum = sum / len(temp)
    return sum


def frequency_score(temp):
    repo = [temp]
    sum = 0

    '''for l in range(4, len(temp)):
        for i in range(0, len(temp)-l+1):
            substr = temp[i:i + l]
            # print(substr)
            repo.append(substr)
    '''
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


def markov_score(temp):
    global model_data
    # query = temp
    model_mat = model_data['mat']
    # threshold = model_data['thresh']
    sum = (gib_detect_train.avg_transition_prob(temp, model_mat))

    return sum


def regularity_score(temp):
    # The search function returns a list of all words that are less than the given
    # maximum distance from the target word
    def search(word, maxCost):

        # build first row
        currentRow = range(len(word) + 1)

        results = []

        # recursively search each branch of the trie
        for letter in trie.children:
            searchRecursive(trie.children[letter], letter, word, currentRow,
                            results, maxCost)

        return results

    # This recursive helper is used by the search function above. It assumes that
    # the previousRow has been filled in already.
    def searchRecursive(node, letter, word, previousRow, results, maxCost):

        columns = len(word) + 1
        currentRow = [previousRow[0] + 1]

        # Build one row for the letter, with a column for each letter in the target
        # word, plus one for the empty string at column 0
        for column in range(1, columns):

            insertCost = currentRow[column - 1] + 1
            deleteCost = previousRow[column] + 1

            if word[column - 1] != letter:
                replaceCost = previousRow[column - 1] + 1
            else:
                replaceCost = previousRow[column - 1]

            currentRow.append(min(insertCost, deleteCost, replaceCost))

        # if the last entry in the row indicates the optimal cost is less than the
        # maximum cost, and there is a word in this trie node, then add it.
        if currentRow[-1] <= maxCost and node.word is not None:
            results.append((node.word, currentRow[-1]))

        # if any entries in the row are less than the maximum cost, then
        # recursively search each branch of the trie
        if min(currentRow) <= maxCost:
            for letter in node.children:
                searchRecursive(node.children[letter], letter, word, currentRow,
                                results, maxCost)

    def normalize(length):
        if length <= 2:
            return 0

        return length.bit_length() - 1


    TARGET = temp
    MAX_COST = normalize(len(TARGET))
    results = search(TARGET, MAX_COST)
    return len(results)


def word2vec_score(temp):
    w1 = [temp]
    try:
        value = model.wv.most_similar(positive=w1, topn=1)[0][1]
        return value
    except Exception as e:
        return  0.0


def correlation_score(temp):
    repo1 = []
    repo2 = []
    sum = 0
    for l in range(2, len(temp) - 1):
        substr1 = temp[:l]
        substr2 = temp[l:]
        # print(substr1, substr2)
        repo1.append(substr1)
        repo2.append(substr2)

    for w1, w2 in zip(repo1, repo2):
        if w1 not in D:
            continue
        if w2 not in D[w1]:
            continue
        # print(w1, w2, D[w1][w2])
        sum = sum   +  D[w1][w2] +  D[w2][w1]

    return sum


# build the trie
trie = TrieNode()
for word in open(DICTIONARY, "rt").read().split():
    WordCount += 1
    trie.insert(word)

# train word2vec
model.train(documents, total_examples=len(documents), epochs=20)
# save only the word vectors
# model.wv.save("vectors/default")

# build correlation map
D = {}
for line in documents[0: 1000]:
    for w in line:
        for x in line:
            if x == w:
                continue
            if w not in D:
                D[w] = {}
            if x not in D[w]:
                D[w][x] = 0
            D[w][x] = D[w][x] + 1


general_path = ''
txt_files = ['conflicker.txt', 'torpig.txt', 'srizbi.txt',
          '500kl1', '500kl2.txt', '9ml1.txt', 'benign.txt']

#general_path2 = '/home/ashiq/PycharmProjects/TensorFlow_MLP_test/dga_dataset/'
dga_files = ['conflicker.txt', 'torpig.txt', 'srizbi.txt',
          '500kl1.txt', '500kl2.txt', '9ml1.txt']

safe_files = ['benign.txt']
for i in range(0, len(txt_files)-1):

    write_file = open( 'out' + txt_files[i], 'w')
    read_file = open( dga_files[i], 'r')
    lines = read_file.readlines()
    count = 0
    for s in lines:
        print(i,count)
        # print(s)
        j = 0
        for j in range(0, len(s)):
            if s[j] == '.':
                break
        string = s[0: j]
        try:
            length = len(string)
            # print(length, end=' ')
            vc = vowel_consonant_ratio(string)
            # print(vc, end=' ')
            four_gram = four_gram_score(string)
            # print(four_gram, end=' ')
            meaning = meaning_score(string)
            # print(meaning, end=' ')
            frequency = frequency_score(string)
            # print(frequency, end=' ')
            markov = markov_score(string)
            # print(markov, end=' ')
            correlation = correlation_score(string)
            # print(correlation, end=' ')
            word2vec = word2vec_score(string)
            # print(word2vec, end=' ')
            regularity = regularity_score(string)
            # print(regularity)

        except Exception as e:
            print(e)
            length = 0
            vc = 0
            four_gram = 0
            meaning = 0
            frequency = 0
            markov = 0
            correlation = 0
            word2vec = 0
            regularity = 0

        final_ans = str(length)+" "+str(vc)+" "+str(four_gram)+" "+str(meaning)+" "+str(frequency)
        final_ans = final_ans + " " + str(markov) + " " + str(correlation) + " "
        final_ans = final_ans + str(regularity) + " " + str(word2vec) + " 1"  # change class [0 for benign, 1 for mal]
        write_file.write(final_ans)

        write_file.write("\n")
        # print(final_ans)
        count = count + 1
        if count == 2000:  # change line no
            break
    write_file.close()

read_file = open( safe_files[0], 'r')
lines = read_file.readlines()
count = 0
for s in lines:
    j = 0
    for j in range(0, len(s)):
        if s[j] == '.':
            break
    string = s[0:j]
    try:
        length = len(string)
        vc = vowel_consonant_ratio(string)
        four_gram = four_gram_score(string)
        meaning = meaning_score(string)
        frequency = frequency_score(string)
        markov = markov_score(string)
        correlation = correlation_score(string)
        word2vec = word2vec_score(string)
        regularity = regularity_score(string)

    except Exception as e:
        length = 0
        vc = 0
        four_gram = 0
        meaning = 0
        frequency = 0
        markov = 0
        correlation = 0
        word2vec = 0
        regularity = 0

    final_ans = str(length)+" "+str(vc)+" "+str(four_gram)+" "+str(meaning)+" "+str(frequency)
    final_ans = final_ans + " " + str(markov) + " " + str(correlation) + " "
    final_ans = final_ans + str(regularity) + " " + str(word2vec) + " 0"  # change class [0 for benign, 1 for mal]
    for i in range(0, len(txt_files)-1):
        write_file = open('out' + txt_files[i], 'a')
        write_file.write(final_ans)
        write_file.write("\n")
        write_file.close()

    #
    # print(count, s)
    count = count + 1
    if count == 2000:  # change line no
        break