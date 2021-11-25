# D:\download\anaconda\envs\ner\python.exe
# -*- coding:utf-8 -*-
# time: 2021/11/25

import os
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

# https://github.com/luopeixiang/named_entity_recognition/blob/master/data.py
def build_corpus(split, make_vocab=True, data_dir='Dataset/weiboNER'):
    """Read in data"""

    assert split in ['train', 'dev', 'test']

    char_lists = [] #
    tag_lists = []

    with open(os.path.join(data_dir, 'weiboNER_2nd_conll.'+split), 'r', encoding='utf-8') as f:
        char_list = []
        tag_list = []
        for line in f:
            if line != '\n':
                word, tag = line.strip('\n').split()
                char_list.append(word[0])
                tag_list.append(tag)
            else:
                char_lists.append(char_list)
                tag_lists.append(tag_list)
                char_list = []
                tag_list = []

    # long sentences first, short sentences last
    char_lists = sorted(char_lists, key=lambda x: len(x), reverse=True)
    tag_lists = sorted(tag_lists, key=lambda x: len(x), reverse=True)

    if make_vocab:  # only for training set
        char2id = build_map(char_lists)
        tag2id = build_map(tag_lists)
        char2id['<UNK>'] = len(char2id)  # unknown char
        char2id['<PAD>'] = len(char2id)  # padding char
        tag2id['<PAD>'] = len(tag2id)  # padding label
        return char_lists, tag_lists, char2id, tag2id
    else:
        return char_lists, tag_lists


def build_map(lists):
    """
    list to id
    returns maps = {name:id}
    """
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)
    return maps

char_lists, tag_lists, char2id, tag2id = build_corpus('train', make_vocab=True, data_dir='Dataset/weiboNER')
print(char_lists)
print(tag_lists)
print(char2id)
print(tag2id)



