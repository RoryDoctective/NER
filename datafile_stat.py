

def stats(path):
    with open(path, 'r', encoding='utf-8') as f:
        sentence_num = 0
        character_num = 0
        char_list = []
        tag_dict ={}
        for line in f:
            if line != '\n':
                try:
                    word, tag = line.strip('\n').split()

                    character_num += 1
                    if word[0] not in char_list:
                        char_list.append(word[0])
                    if tag not in tag_dict:
                        tag_dict[tag] = 0
                    else:
                        tag_dict[tag] += 1
                except: # the word is empty
                    tag = line.strip('\n').split()
                    character_num += 1
                    if " " not in char_list:
                        char_list.append(word[0])
                    if "kshdfuwehf" not in tag_dict:
                        tag_dict["kshdfuwehf"] = 0
                    else:
                        tag_dict["kshdfuwehf"] += 1



            else:  # line = \n end of a sentence
                sentence_num += 1
    distinct_char = len(char_list)
    distinct_tag = len(tag_dict)
    return sentence_num, character_num, distinct_char, distinct_tag, tag_dict


# NAME = ["CLUENER", "E-commerce", "Finance", "Literature", "MSRA","Novel", "PeoplesDaily","Resume","Weibo"]

if __name__ == "__main__":
    print("Start statistic of datasets ")

    NAME = ["CLUENER",
            "E-commerce",
            "Finance",
            "Literature",
            "MSRA",
            # "Multimodal", # need preprocessing
            "Novel",
            "PeoplesDaily",
            "Resume",
            "Weibo"]

    for i in range(len(NAME)):
        path_train = f"Dataset/{NAME[i]}/demo.train"
        path_dev = f"Dataset/{NAME[i]}/demo.dev"
        path_test = f"Dataset/{NAME[i]}/demo.test"
        paths = [path_train, path_dev, path_test]

        print("________________________")
        print(f"Dataset:{NAME[i]}")
        set = ["train","dev","test"]

        for j in range(len(paths)):
            sentence_num, character_num, distinct_char, distinct_tag, tag_dict = stats(paths[j])
            print("__________")
            print(f"{set[j]}:")
            print(f"number of sentences: {sentence_num}")
            print(f"number of characters: {character_num}")
            print(f"number of distinct characters: {distinct_char}")
            print(f"number of distinct tags: {distinct_tag}")
            print(f"tags statics:")
            tag_dict_key_list = []
            for key in tag_dict.keys():
                tag_dict_key_list.append(key)
            for k in range(len(tag_dict_key_list)):
                print(f"{tag_dict_key_list[k]}:{tag_dict[tag_dict_key_list[k]]}")

