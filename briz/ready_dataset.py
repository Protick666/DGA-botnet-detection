import csv


def conversion():
    with open('scores.txt', 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line.split(" ") for line in stripped if line)
        print(lines)
        with open('../input_csv_files/other/kraken.csv', 'w') as out_file: # change file path
            writer = csv.writer(out_file)
            writer.writerow(('length', 'vowel-consonant_ratio', '4-gram score', 'dict_score', 'corpus_score',
                             'markov_score'))
            writer.writerows(lines)


def label_correction():
    with open('allsafe_scores.txt', 'r') as in_file:
        lines = in_file.readlines()
        for line in lines:
            #print(line)
            line = line.strip()
            a, b, c, d, e, f, g = line.split(' ')
            #print(a, b, c, d, e)
            g = "0"
            with open('output_allsafe_scores.txt', 'a') as out_file:
                line = ' '.join([a, b, c, d, e, f, g])
                out_file.write(line+'\n')
        #lines[4] = int(lines[4])-1


label_correction()
# conversion()