import os    
    
data_dir = "only_ch.txt"
count = 0
with open(data_dir, 'r', encoding='utf-8') as f:
    for line in f:  # for each line
        # print(line)
        # print(line.strip('\n').split())
        word, attribute= line.strip('\n').split()[:4]
        #print(word)
        count += 1
        if count >= 100:
            exit()
            