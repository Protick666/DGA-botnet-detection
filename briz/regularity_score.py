#!/usr/bin/python
#By Steve Hanov, 2011. Released to the public domain
import time
import sys

DICTIONARY = "dicti.txt"
#outF = open("/home/ashiq/Desktop/temp_regularity_Scores/pcfg_ipv4_num", "w")
# TARGET = ""
# MAX_COST = 0

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

    def insert( self, word ):
        node = self
        for letter in word:
            if letter not in node.children:
                node.children[letter] = TrieNode()

            node = node.children[letter]

        node.word = word

# read dictionary file into a trie
trie = TrieNode()
for word in open(DICTIONARY, "rt").read().split():
    WordCount += 1
    trie.insert( word )

#print("Read %d words into %d nodes" % (WordCount, NodeCount))

# The search function returns a list of all words that are less than the given
# maximum distance from the target word
def search( word, maxCost ):

    # build first row
    currentRow = range( len(word) + 1 )

    results = []

    # recursively search each branch of the trie
    for letter in trie.children:
        searchRecursive( trie.children[letter], letter, word, currentRow,
            results, maxCost )

    return results

# This recursive helper is used by the search function above. It assumes that
# the previousRow has been filled in already.
def searchRecursive( node, letter, word, previousRow, results, maxCost ):

    columns = len( word ) + 1
    currentRow = [ previousRow[0] + 1 ]

    # Build one row for the letter, with a column for each letter in the target
    # word, plus one for the empty string at column 0
    for column in range( 1, columns ):

        insertCost = currentRow[column - 1] + 1
        deleteCost = previousRow[column] + 1

        if word[column - 1] != letter:
            replaceCost = previousRow[ column - 1 ] + 1
        else:
            replaceCost = previousRow[ column - 1 ]

        currentRow.append( min( insertCost, deleteCost, replaceCost ) )

    # if the last entry in the row indicates the optimal cost is less than the
    # maximum cost, and there is a word in this trie node, then add it.
    if currentRow[-1] <= maxCost and node.word is not None:
        results.append( (node.word, currentRow[-1] ) )

    # if any entries in the row are less than the maximum cost, then
    # recursively search each branch of the trie
    if min(currentRow) <= maxCost:
        for letter in node.children:
            searchRecursive( node.children[letter], letter, word, currentRow,
                results, maxCost )


def normalize(length):
    if length <= 2:
        return 0

    return length.bit_length() - 1



off = ["pcfg_dict_regu_scores.txt", "pcfg_dict_num_regu_scores.txt", "pcfg_ipv4_regu_scores.txt" , "pcfg_ipv4_num_regu_scores.txt", "srizbi_regu_scores.txt", "torpig_regu_scores.txt", "zeus_regu_scores.txt", "kraken_regu_scores.txt", "DNL1_regu_scores.txt", "DNL2_regu_scores.txt", "DNL3_regu_scores.txt", "DNL4_regu_scores.txt", "500KL1_regu_scores.txt", "500KL2_regu_scores.txt", "500KL3_regu_scores.txt", "9ML1_regu_scores.txt"];

iff = ["pcfg_dict.txt", "pcfg_dict_num.txt", "pcfg_ipv4.txt" , "pcfg_ipv4_num.txt", "srizbi.txt", "torpig.txt", "zeus.txt", "kraken.txt", "DNL1.txt", "DNL2.txt", "DNL3.txt", "DNL4.txt", "500KL1.txt", "500KL2.txt", "500KL3.txt", "9ML1.txt"];


for it in range(0,16):
        #bad_repo = {}
        print(it)
        #model_data = pickle.load(open('gib_model.pki', 'rb'))
        #dct = enchant.Dict("en_US")
        outF = open(off[it],"w")


        with open(iff[it]) as f:  # change file name
            lines = f.readlines()
            count = 0
            for s in lines:
                print(count)
                j = 0
                for j in range(0,len(s)):
                    if s[j] == '.':
                        break

                string = s[0:j]

                try:
                    TARGET = string
                    MAX_COST = normalize(len(TARGET))
                    # start = time.time()
                    results = search(TARGET, MAX_COST)
                    # end = time.time()

                    # for result in results: print(result)
                    final_ans = len(results)

                    # print("Search took %g s" % (end - start))

                except Exception as e:
                    final_ans = 0

                final_ans = str(final_ans) + " 1"  # change class [0 for benign, 1 for mal]
                outF.write(final_ans)

                outF.write("\n")
                #print(count, s)
                count = count+1
                if count == 2000:  # change line no
                    break

        outF.close()

