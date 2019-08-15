import csv
txt_files = ['conflicker.txt', 'torpig.txt', 'srizbi.txt',
          '500kl1', '500kl2.txt', '9ml1.txt']
#conflicker, torpig, srizbi, 500kl1, 500kl2, zeus, 9ml1
def conversion():
    for i in range(0, len(txt_files) - 1):
        #write_file = open('out' + txt_files[i], 'w')
        with open( 'out' + txt_files[i], 'r') as in_file:
            stripped = (line.strip() for line in in_file)
            lines = (line.split(" ") for line in stripped if line)
            # print(type(lines))
            with open(txt_files[i][0:-4] + '.csv', 'w') as out_file:  # change file path
                writer = csv.writer(out_file)
                writer.writerow(('length', 'vowel-consonant_ratio', '4-gram score', 'meaning_score', 'freq_score',
                                 'markov_score', 'correlation_score', 'reg_score', 'w2v_score', 'class'))
                writer.writerows(lines)





def label_correction():
    with open('all_.txt', 'r') as in_file:
        lines = in_file.readlines()
        for line in lines:
            #print(line)
            line = line.strip()
            a, b, c, d, e = line.split(' ')
            #print(a, b, c, d, e)
            e = str(int(e) - 1)
            with open('output.txt', 'a') as out_file:
                line = ' '.join([a, b, c, d, e])
                out_file.write(line+'\n')
        #lines[4] = int(lines[4])-1


# label_correction()
conversion()