# adapt from the chinese_t2s.py

import sys
import os
from optparse import OptionParser


class C2R(object):
    def __init__(self, infile, outfile):
        self.infile = infile
        self.outfile = outfile

        self.c_corpus = []
        self.r_corpus = []
        self.char_2_radical= {}
        self.rad_index_2_rad = {}

        self.get_one_radical()  # read radicals
        self.radindex_and_rad() # read index radicals
        self.read(self.infile)  # read lines
        self.c2r()              # convert
        self.write(self.r_corpus, self.outfile)  # write out

    def read(self, path):
        if os.path.isfile(path) is False:
            print("path is not a file")
            exit()
        now_line = 0
        with open(path, encoding="UTF-8") as f:
            for line in f:
                now_line += 1
                line = line.replace("\n", "").replace("\t", "")
                self.c_corpus.append(line)
        print("read finished")

    def write(self, list, path):
        print("writing now......")
        if os.path.exists(path):
            os.remove(path)
        file = open(path, encoding="UTF-8", mode="w")
        for line in list:
            file.writelines(line + "\n")
        file.close()
        print("writing finished.")
    
    def c2r(self):
        now_line = 0
        all_line = len(self.c_corpus)
        for line in self.c_corpus:
            now_line += 1
            if now_line % 1000 == 0:
                sys.stdout.write("\rhandling with the {} line, all {} lines.".format(now_line, all_line))
            self.r_corpus.append(self.convert(line))
        sys.stdout.write("\rhandling with the {} line, all {} lines.".format(now_line, all_line))
        print("\nhandling finished")

    def convert(self, line):
        # input "双方都是"
        # output "一二三四"
        str_of_rad = ""
        lst_of_chr = self.split(line)
        # get all radical
        for i in range(len(lst_of_chr)):
            char = lst_of_chr[i]
            its_radin = self.char_2_radical[char]
            the_rad= self.rad_index_2_rad[int(its_radin)]
            str_of_rad += the_rad
        return str_of_rad

    
    def split(self, word):
        return [char for char in word]

    def get_one_radical(self, data_dir='../Radical/Unihan_IRGSources.txt'):
        with open(data_dir, 'r', encoding='utf-8') as f:
            for line in f:  # for each line
                # print(line.strip('\n').split())
                word, attribute, content = line.strip('\n').split()[:3]
                if attribute == "kRSUnicode":
                    char = chr(int(word[2:], 16))
                    if "." in content:
                        radical, stroke = content.split(".")
                    else:
                        radical = content
                        stroke = '0'
                    radical = radical.strip("'")
                    self.char_2_radical[char]=radical

    def radindex_and_rad(self, data_dir='../Radical/radindex_rad.txt'):
        with open(data_dir, 'r', encoding='utf-8') as f:
            index = 0
            for line in f:  # for each line
                index += 1
                radical, _ = line.strip('\n').split()[:2]
                self.rad_index_2_rad[index]=radical


if __name__ == "__main__":
    print("chinese character to radical")
    # input = "./wiki_zh_10.txt"
    # output = "wiki_zh_10_sim.txt"
    # T2S(infile=input, outfile=output)

    parser = OptionParser()
    parser.add_option("--input", dest="input", default="", help="traditional file")
    parser.add_option("--output", dest="output", default="", help="simplified file")
    (options, args) = parser.parse_args()

    input = options.input
    output = options.output

    try:
        C2R(infile=input, outfile=output)
        print("All Finished.")
    except Exception as err:
        print(err)