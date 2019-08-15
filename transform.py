for it in range(0,1):
        #bad_repo = {}
        #model_data = pickle.load(open('gib_model.pki', 'rb'))
        #dct = enchant.Dict("en_US")
        outF = open("brizlal er dadur kabin.txt","w")


        with open("brizsapex.txt") as f:  # change file name
            lines = f.readlines()
            count = 0
            for s in lines:
                p = s.split()
                #print(p[1])
                #print(sc, string)

                #final_ans = str(sc) + " 1"  # change class [0 for benign, 1 for mal]
                outF.write(p[1])

                outF.write("\n")
                #print(count, s)
                count = count+1
                if count == 2000:  # change line no
                    break

        outF.close()