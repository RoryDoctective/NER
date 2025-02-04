# D:\download\anaconda\envs\ner\python.exe
# -*- coding:utf-8 -*-
# time: 2022/3/2
# the codes are learnt from https://github.com/shouxieai/nlp-bilstm_crf-ner

import os
from itertools import zip_longest
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
import textwrap
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.bayesopt import BayesOptSearch

# global:
# Weibo, Resume, MSRA(no_dev), Literature(error), CLUENER, Novel(long_time_to_test), Finance(no_dev), E-commerce(error)
DATASET = 'Weibo'
DEV = True

REMOVE_O = True
SHOW_REPORT = True
DRAW_GRAPH = True

BI_LSTM_CRF = True

One_Radical = False
Three_Radicals = False


# https://github.com/luopeixiang/named_entity_recognition/blob/master/data.py
def build_corpus(split, make_vocab=True, data_dir='Dataset/Weibo'):
    """Read in data"""

    assert split in ['train', 'dev', 'test']

    char_lists = []
    tag_lists = []

    with open(os.path.join(data_dir, 'demo.' + split), 'r', encoding='utf-8') as f:
        char_list = []
        tag_list = []
        for line in f:
            if line != '\n':
                try:
                    word, tag = line.strip('\n').split()
                    char_list.append(word[0])
                    tag_list.append(tag)
                except:
                    # to stop
                    word, tag = line.strip('\n').split()
                    #
                    tag = line.strip('\n').split()
                    char_list.append(' ')
                    tag_list.append(tag[0])
                    # print(tag)

            else:  # line = \n
                if BI_LSTM_CRF:
                    char_lists.append(char_list + ["<END>"])
                    tag_lists.append(tag_list + ["<END>"])
                else:
                    char_lists.append(char_list)
                    tag_lists.append(tag_list)
                char_list = []
                tag_list = []

    # reverse = False == shortest sentences first, longest sentences last
    # when do LSTM-CRF, set it to True
    char_lists = sorted(char_lists, key=lambda x: len(x), reverse=True)
    tag_lists = sorted(tag_lists, key=lambda x: len(x), reverse=True)

    if make_vocab:  # only for training set
        char2id = build_map(char_lists)
        tag2id = build_map(tag_lists)
        char2id['<UNK>'] = len(char2id)  # unknown char
        char2id['<PAD>'] = len(char2id)  # padding char
        tag2id['<PAD>'] = len(tag2id)  # padding label

        if BI_LSTM_CRF:
            char2id['<START>'] = len(char2id)  # start char
            tag2id['<START>'] = len(tag2id)  # start label

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


def build_one_radical(data_dir='Radical/Unihan_IRGSources.txt'):
    """Read in data"""
    global char_to_index
    char_radical_stroke_list = []

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
                char_radical_stroke_list.append([char, radical, stroke])
    # a = [[list[0], list[2]] for index, list in enumerate(char_radical_stroke_list)]

    # get [[id, radical]]
    id_radical = [[char_to_index.get(list[0], char_to_index["<UNK>"]), list[1]] for index, list in
                  enumerate(char_radical_stroke_list)]

    # deal with multiple <UNK>
    id_radical_list = []
    for i in range(len(id_radical)):
        if id_radical[i][0] != char_to_index["<UNK>"]:
            id_radical_list.append(id_radical[i])

    # deal with <UNK, PAD, START>
    id_radical_list.append([char_to_index["<UNK>"], '0'])

    # sort by index
    id_radical_list = sorted(id_radical_list, key=itemgetter(0))

    # add 0 as the radical for things not a chinese character
    for i in range(len(char_to_index)):
        if len(id_radical_list) > i:
            if id_radical_list[i][0] != i:
                id_radical_list.append([i, '0'])
                id_radical_list = sorted(id_radical_list, key=itemgetter(0))
        else:
            id_radical_list.append([i, '0'])
            id_radical_list = sorted(id_radical_list, key=itemgetter(0))

    # get only the radical, and make the map
    ordered_radical = []
    for i in range(len(id_radical_list)):
        ordered_radical.append(int(id_radical_list[i][1]))
    # this is the id2radical
    return ordered_radical


def dummy_radical():
    global char_to_index
    dummy = []
    for i in range(len(char_to_index)):
        dummy.append(0)
    return dummy, 0


def build_ids(data_dir='Radical/CHISEids.txt'):
    """
        Read in data
        id_to_many_radicals -> [id, ’⬚十日‘]
        'U+0080'*# = padding
        'itself'*3 = not chinese # TODO
    """
    global char_to_index, id_to_char

    # read in
    char_radicals_list = []
    with open(data_dir, 'r', encoding='utf-8') as f:
        for line in f:  # for each line
            # print(line.strip('\n').split())
            unicode, character, component = line.strip('\n').split()[:3]
            component = component.strip("[]GTJKVH'")
            char_radicals_list.append([chr(int(unicode[2:], 16)), component])

    # get [[id, radicals]]
    id_radicals = [[char_to_index.get(list[0], char_to_index["<UNK>"]), list[1]]
                   for index, list in enumerate(char_radicals_list)]

    # deal with multiple <UNK>
    id_radicals_list = []
    for i in range(len(id_radicals)):
        if id_radicals[i][0] != char_to_index["<UNK>"]:
            id_radicals_list.append(id_radicals[i])

    # deal with <UNK, PAD, START> using 0 as unknow
    id_radicals_list.append([char_to_index["<UNK>"], chr(int('0080', 16)) * 3])

    # sort by index
    id_radicals_list = sorted(id_radicals_list, key=itemgetter(0))

    # add 'itself' as the radical for things not a chinese character
    for i in range(len(char_to_index)):
        if len(id_radicals_list) > i:
            if id_radicals_list[i][0] != i:
                id_radicals_list.append([i, id_to_char[i] * 3])
                id_radicals_list = sorted(id_radicals_list, key=itemgetter(0))
        else:
            id_radicals_list.append([i, id_to_char[i] * 3])
            id_radicals_list = sorted(id_radicals_list, key=itemgetter(0))

    # get only the radical, and make the map
    ordered_radicals = []
    for i in range(len(id_radicals_list)):
        ordered_radicals.append(id_radicals_list[i][1])

    # deal with PAD, START
    if ordered_radicals[-1] == '<START>':
        ordered_radicals[-1] = chr(int('0080', 16)) * 3
    if ordered_radicals[-1] == '<PAD>':
        ordered_radicals[-1] = chr(int('0080', 16)) * 3

    # Aiming at 1 ⬚ + 2 radicals:
    formal_radicals = []
    for i in range(len(ordered_radicals)):
        radical_list = ordered_radicals[i]

        # means it is not a chinese character/ it is a padding
        if len(radical_list) == 3:
            if radical_list[0] == radical_list[1] and radical_list[1] == radical_list[2]:
                formal_radicals.append(radical_list)
                continue

        # finding the first idc, add it
        # finding the first 2 dc, add it
        idc = 0
        dc = 0
        temp_idc = []
        temp_dc = []
        for j in range(len(radical_list)):
            if is_idc(radical_list[j]) and idc < 1:
                temp_idc.append(radical_list[j])
                idc += 1
            elif not is_idc(radical_list[j]) and dc < 2:
                temp_dc.append(radical_list[j])
                dc += 1
        if len(temp_idc) < 1:
            temp_idc.append('⬚')
        if len(temp_dc) == 1:
            temp_dc.append(temp_dc[0])
        elif len(temp_dc) == 0:
            temp_dc.append(chr(int('0080', 16)))
            temp_dc.append(chr(int('0080', 16)))
        temp_idc.extend(temp_dc)
        formal_radicals.append(temp_idc)

    # get all the possible components:
    all_rad = []
    for i in range(len(formal_radicals)):
        for j in range(3):
            if formal_radicals[i][j] not in all_rad:
                all_rad.append(formal_radicals[i][j])

    # radical 2 radical ids
    rad3toradid = build_map(all_rad)

    finnal_id_to_id = []
    for i in range(len(formal_radicals)):
        temp = []
        for j in range(3):
            temp.append(rad3toradid[formal_radicals[i][j]])
        finnal_id_to_id.append(temp)
    return finnal_id_to_id, len(rad3toradid)


def is_idc(text):
    """
    Return True if between "⿰"(U + 2FF0) to"⿻"(U + 2FFB)
    """
    low = int('2FF0', 16)
    up = int('2FFB', 16)
    text = ord(text)
    if low <= text <= up:
        return True
    else:
        return False


def argmax(vec):
    # vec of size [10, 1, 20]
    # torch.max(torch.tensor([[1, 2, 3, 4, 5], [6, 5, 4, 3, 2]]), 1)
    # Out[14]:
    # torch.return_types.max(
    #     values=tensor([5, 6]),
    #     indices=tensor([4, 0]))

    # return the argmax as a python int
    _, idx = torch.max(vec, 2)

    # size = [10,1]
    return idx


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    # vec of size [10, 1, 20]
    # max_score of size [10,1]
    max_score, _ = torch.max(vec, 2)
    # max_score_broadcast of size [10, 1, 20]
    max_score_broadcast = max_score.view(vec.size()[0], 1, -1).expand(vec.size()[0], 1, vec.size()[2])
    # vec - max_score
    # size = [10, 1]
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast), 2))


class MyDataset(Dataset):  # Inherit the torch Dataset
    # 汉字，标签
    def __init__(self, data, tag, char2id, tag2id, id2rad):
        # char to index/ 汉字变数字
        self.data = data
        self.tag = tag
        self.char2id = char2id
        self.tag2id = tag2id
        self.id2rad = id2rad

    def __getitem__(self, index):
        # get one sentence
        sentence = self.data[index]
        sentence_tag = self.tag[index]

        # get each char's/tag's index in each sentence
        #   tag should always be seen before.
        tag_index = [self.tag2id[i] for i in sentence_tag]
        char_index = []
        for i in sentence:
            if i in self.char2id:
                char_index.append(self.char2id[i])
            else:  # if not in the look up table, it is unknown.
                char_index.append(self.char2id['<UNK>'])

        radical_index = []
        if One_Radical:
            for i in char_index:
                radical_index.append(self.id2rad[i])
        elif Three_Radicals:
            for i in char_index:
                temp = []
                for j in range(3):
                    temp.append(self.id2rad[i][j])
                radical_index.append(temp)
        else:
            for i in char_index:
                radical_index.append(self.id2rad[i])

        return char_index, tag_index, radical_index

    def __len__(self):
        # one char -> one tag
        assert len(self.data) == len(self.tag)
        # define: time of running __getitem__
        return len(self.tag)

    # function for making padding
    # making all sentence inside the batch has the same len
    def pro_batch_data(self, batch_data):
        global device
        sentences = []
        sentences_tag = []
        batch_lens = []
        sentences_radical = []

        for sentence, sentence_tag, sentence_radical in batch_data:
            sentences.append(sentence)
            sentences_tag.append(sentence_tag)
            sentences_radical.append(sentence_radical)
            batch_lens.append(len(sentence))
        batch_max_len = max(batch_lens)

        # add padding
        sentences = [i + [self.char2id['<PAD>']] * (batch_max_len - len(i)) for i in sentences]
        sentences_tag = [i + [self.tag2id['<PAD>']] * (batch_max_len - len(i)) for i in sentences_tag]

        if One_Radical:
            sentences_radical = [i + [self.id2rad[self.char2id['<PAD>']]] * (batch_max_len - len(i))
                                 for i in sentences_radical]
            # return tensor
            return torch.tensor(sentences, dtype=torch.int64, device=device), \
                   torch.tensor(sentences_tag, dtype=torch.int64, device=device), \
                   batch_lens, \
                   torch.tensor(sentences_radical, dtype=torch.int64, device=device)

        elif Three_Radicals:
            sentences_radical = [i + [self.id2rad[self.char2id['<PAD>']]] * (batch_max_len - len(i))
                                 for i in sentences_radical]
            # separate those 3
            Temp = np.array(sentences_radical)
            a, b = Temp[:, :, 0].shape
            # Temp[:,:,0].reshape((a,b,1)
            # return tensor
            return torch.tensor(sentences, dtype=torch.int64, device=device), \
                   torch.tensor(sentences_tag, dtype=torch.int64, device=device), \
                   batch_lens, \
                   [torch.tensor(Temp[:, :, 0], dtype=torch.int64, device=device),
                    torch.tensor(Temp[:, :, 1], dtype=torch.int64, device=device),
                    torch.tensor(Temp[:, :, 2], dtype=torch.int64, device=device)]
        else:  # no radicals
            sentences_radical = [i + [self.id2rad[self.char2id['<PAD>']]] * (batch_max_len - len(i))
                                 for i in sentences_radical]
            return torch.tensor(sentences, dtype=torch.int64, device=device), \
                   torch.tensor(sentences_tag, dtype=torch.int64, device=device), \
                   batch_lens, \
                   torch.tensor(sentences_radical, dtype=torch.int64, device=device)


# model = LSTMModel(char_num, embedding_num, hidden_num, class_num, bi)
class LSTMModel(nn.Module):
    # 多少个不重复的汉字， 多少个embedding，LSTM隐藏大小， 分类类别， 双向
    def __init__(self, char_num, embedding_num, total_rad_ids, hidden_num, class_num, bi=True):
        super().__init__()
        self.embedding = nn.Embedding(char_num, embedding_num)

        # add dropout
        # prob = 0.5 !! can be tunned
        self.drop = nn.Dropout(0.5)

        if One_Radical:
            self.one_radical_embedding = nn.Embedding(total_rad_ids, 50)
            # 一层， batch在前面
            self.lstm = nn.LSTM(embedding_num + 50, hidden_num, num_layers=1, batch_first=True, bidirectional=bi)
        elif Three_Radicals:
            self.one_radical_embedding = nn.Embedding(total_rad_ids, 100)
            self.lstm = nn.LSTM(embedding_num + 100 * 3, hidden_num, num_layers=1, batch_first=True, bidirectional=bi)
        else:
            self.lstm = nn.LSTM(embedding_num, hidden_num, num_layers=1, batch_first=True, bidirectional=bi)

        if bi:  # 双向：hidden# * 2
            self.classifier = nn.Linear(hidden_num * 2, class_num)
        else:
            self.classifier = nn.Linear(hidden_num, class_num)

        self.cross_loss = nn.CrossEntropyLoss()  # it contains softmax inside

    def forward(self, batch_char_index, batch_onerad_index, batch_tag_index=None):
        if One_Radical:
            # if returns [5,4,101] means 5 batches, each batch 4 char, each char 101 dim represent.
            embedding_char = self.embedding(batch_char_index)  # get character embedding
            embedding_onerad = self.one_radical_embedding(batch_onerad_index)  # get radical embedding
            embedding = torch.cat([embedding_char, embedding_onerad], 2)
        elif Three_Radicals:
            embedding_char = self.embedding(batch_char_index)  # get character embedding
            embedding_onerad_0 = self.one_radical_embedding(batch_onerad_index[0])  # get radical embedding
            embedding_onerad_1 = self.one_radical_embedding(batch_onerad_index[1])
            embedding_onerad_2 = self.one_radical_embedding(batch_onerad_index[2])
            embedding = torch.cat([embedding_char, embedding_onerad_0, embedding_onerad_1, embedding_onerad_2], 2)
        else:  # normal
            embedding = self.embedding(batch_char_index)

        # add dropout layer
        embedding = self.drop(embedding)

        # out = 每一个字的结果 = [5,4,214] = 5 batches, each batch 4 char, each char 214 hidden
        # hidden = we don't care
        out, hidden = self.lstm(embedding)
        # pre = 每一个字的预测值 = [5,4,29]= 5 batches, each batch 4 char, each char 29 个可能的prediction
        prediction = self.classifier(out)
        # get 最大值 index = the pre
        # 拉成横条
        # saved in side the model for call
        self.prediction = torch.argmax(prediction, dim=-1).reshape(-1)

        if batch_tag_index is not None:  # for the final test
            # change shape to [batch_size * char_num] * class_num & [batch_size * char_num]
            # prediction.shape = [5,4,29]
            # prediction.reshape(-1, prediction.shape[-1]) == [20， 29]
            # batch_tag_index.reshape(-1) == [20]
            loss = self.cross_loss(prediction.reshape(-1, prediction.shape[-1]), batch_tag_index.reshape(-1))
            return loss


# model = LSTM_CRF_Model(char_num, embedding_num, hidden_num, class_num, bi, tag_to_id)
class LSTM_CRF_Model(nn.Module):
    # 多少个不重复的汉字， 多少个embedding，LSTM隐藏大小， 分类类别， 双向
    def __init__(self, char_num, embedding_num, total_rad_ids, hidden_num, class_num, bi=True):
        super().__init__()
        # self.tag_to_id = tag_to_id
        # self.tagset_size = class_num

        # add dropout
        # prob = 0.5 !! can be tunned
        self.drop = nn.Dropout(0.5)

        # 每一个汉字 + 一个embedding长度 = embedding
        self.embedding = nn.Embedding(char_num, embedding_num)

        if One_Radical:
            self.one_radical_embedding = nn.Embedding(total_rad_ids, 50)
            # 一层， batch在前面
            self.lstm = nn.LSTM(embedding_num + 50, hidden_num, num_layers=1, batch_first=True, bidirectional=bi)
        elif Three_Radicals:
            self.one_radical_embedding = nn.Embedding(total_rad_ids, 100)
            self.lstm = nn.LSTM(embedding_num + 100 * 3, hidden_num, num_layers=1, batch_first=True, bidirectional=bi)
        else:  # no radical
            # 一层， batch在前面
            self.lstm = nn.LSTM(embedding_num, hidden_num, num_layers=1, batch_first=True, bidirectional=bi)

        # Maps the output of the LSTM into tag space.
        # = hidden2tag
        if bi:  # 双向：hidden# * 2
            self.classifier = nn.Linear(hidden_num * 2, class_num)
        else:
            self.classifier = nn.Linear(hidden_num, class_num)

        # here comes the difference:
        # A kind of Tensor that is to be considered a module parameter
        # eg: if class_num = 4, then nn.Parameter() =
        # tensor([[0.2500, 0.2500, 0.2500, 0.2500],
        #         [0.2500, 0.2500, 0.2500, 0.2500],
        #         [0.2500, 0.2500, 0.2500, 0.2500],
        #         [0.2500, 0.2500, 0.2500, 0.2500]])
        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j. (j -> i)
        self.transition = nn.Parameter(torch.ones(class_num, class_num) * 1 / class_num)
        global tag_to_id
        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transition.data[tag_to_id['<START>'], :] = -10000
        self.transition.data[:, tag_to_id['<END>']] = -10000
        self.transition.data[tag_to_id['<PAD>'], tag_to_id['<END>']] = 0.05

    # emission score
    def _get_lstm_features(self, batch_data, batch_onerad):
        # embedding
        if One_Radical:
            # if returns [5,4,101] means 5 batches, each batch 4 char, each char 101 dim represent.
            embedding_char = self.embedding(batch_data)  # get character embedding
            embedding_onerad = self.one_radical_embedding(batch_onerad)  # get radical embedding
            embedding = torch.cat([embedding_char, embedding_onerad], 2)
        elif Three_Radicals:
            embedding_char = self.embedding(batch_data)  # get character embedding
            embedding_onerad_0 = self.one_radical_embedding(batch_onerad[0])  # get radical embedding
            embedding_onerad_1 = self.one_radical_embedding(batch_onerad[1])
            embedding_onerad_2 = self.one_radical_embedding(batch_onerad[2])
            embedding = torch.cat([embedding_char, embedding_onerad_0, embedding_onerad_1, embedding_onerad_2], 2)
        else:  # normal
            embedding = self.embedding(batch_data)
        # add dropout layer
        embedding = self.drop(embedding)
        # do bi-lstm
        out, _ = self.lstm(embedding)
        # value of each tag
        emission = self.classifier(out)
        # [10, 176, 20] = batch, max_length_of_input_sentences, tags的种类
        return emission

    # total path score
    def _forward_alg(self, feats):
        device = feats.device

        # [10, 176, 20] = batch, max_length_of_input_sentences, tags的种类
        batch_size, max_len, out_size = feats.size()
        # Do the forward algorithm to compute the partition function (all path score)
        # size = [batch_size, len(tags)], value = -10000
        init_alphas = torch.full((batch_size, 1, out_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[:, :, tag_to_id['<START>']] = 0.

        # Wrap in a variable so that we will get automatic backprop
        # [10, 1, 20]
        forward_var = init_alphas.to(device)

        # swap dimension of the feats
        # [10, 176, 20] -> [176, 10, 20]
        feats = torch.transpose(feats, 0, 1)
        # Iterate through the sentence
        for feat in feats:  # feat size = [10,20]
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(out_size):
                # broadcast the emission score: it is the same regardless of the previous tag
                # size of [10] -> [10,1,1] -> [10, 1, 20]
                emit_score = feat[:, next_tag].view(batch_size, 1, -1).expand(batch_size, 1, out_size)
                # the ith entry of trans_score = the score of transitioning to next_tag from i
                # i(all i s) -> next_tag
                # size [20] ->  [10, 1, 20]
                trans_score = self.transition[next_tag].expand(batch_size, 1, out_size)
                # The ith entry of next_tag_var = the value for the edge (i -> next_tag) before we do log-sum-exp
                # size = [10, 1, 20]
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag = log-sum-exp of all the scores.
                # size = [10,1] # TODO
                alphas_t.append(log_sum_exp(next_tag_var).view(batch_size, -1))
            # size [10,1,20]
            forward_var = torch.cat(alphas_t, 1).view(batch_size, 1, out_size)
        """to end is not needed"""
        # # i (all s) -> STOP
        # terminal_var = forward_var + self.transition[tag_to_id['<END>']].expand(batch_size, 1, out_size)
        # final
        terminal_var = forward_var
        alpha = log_sum_exp(terminal_var)
        # size = [10, 1]
        return alpha

    # real path score
    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        # feats = size [10, 176, 20]
        # [10, 176, 20] = batch, max_length_of_input_sentences, tags的种类
        batch_size, max_len, out_size = feats.size()
        device = feats.device
        # tags = size [10, 176]
        # score = size [10]
        _score = torch.zeros(10).to(device)
        # [tag_to_id['<START>']] = [19]
        # torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long) = tensor([19])
        # [torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags] =
        #       [tensor([3]), tensor([0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 2])]
        # tags =
        #       tensor([3, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 2])

        # add start
        #   torch.Size([10, 177])
        tags = torch.cat([torch.tensor([tag_to_id['<START>']], dtype=torch.long).expand(batch_size, 1).to(device), tags], 1)

        # swap dimension of the feats
        # [10, 176, 20] -> [176, 10, 20]
        feats = torch.transpose(feats, 0, 1)

        # do middle
        # # feats = size [176, 10, 20]
        # feat = [10,20] ; i = 0-175
        for i, feat in enumerate(feats):
            # emission
            emission = []
            for j in range(batch_size):  # TODO
                emission.append(feat[j][tags[:, i + 1][j]].view(1,-1))
            # size = [10,1]
            emission = torch.cat(emission, 0).reshape(-1)

            # Trans: tags[i] -> tags[i + 1] ; size[10]
            # + Emission size[10]
            _score = _score + \
                self.transition[tags[:, i + 1], tags[:, i]] + emission

        """this end tag is not needed"""
        # # add end : tran: 2->4, last tag to end tag
        # _score = _score + self.transition[torch.tensor(tag_to_id['<END>'], dtype=torch.long).expand(batch_size), tags[:, -1]]

        # size[10, 1]
        return _score.view(-1, 1)

    # return loss / DONE
    def forward(self, batch_data, batch_onerad, batch_tag):
        # value of each tag
        # [10, 176, 20] = batch, max_length_of_input_sentences, tags的种类
        emission = self._get_lstm_features(batch_data, batch_onerad)
        batch_size, max_len, out_size = emission.size()

        # alpha = total path score
        # alpha size = [10, 1]
        forward_score = self._forward_alg(emission)

        # gold = real path score
        # gold size = [10,1]
        gold_score = self._score_sentence(emission, batch_tag)

        # loss = #
        # 520, -9980
        # 519 -17
        loss = (forward_score.sum() - gold_score.sum()) / batch_size
        print(loss)
        return loss

    def prediction(self, test_sents_tensor, test_onerad_tensor):
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(test_sents_tensor, test_onerad_tensor)
        # Find the best path, given the features.
        # [10], [10, 176]
        predicted_score, tag_seq = self._viterbi_decode(lstm_feats)
        # e.g. tensor(2.6907), [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
        return predicted_score, tag_seq

    # Find the best path, given the features(emission scores).
    def _viterbi_decode(self, feats):
        """使用维特比算法进行解码"""
        # Gives the score of a provided tag sequence
        # feats = size [10, 176, 20] = 176 char of 20 features
        # [10, 176, 20] = batch, max_length_of_input_sentences, tags的种类
        # B, L, T
        batch_size, max_len, out_size = feats.size()
        global tag_to_id
        start_id = tag_to_id['<START>']
        end_id = tag_to_id['<END>']
        pad = tag_to_id['<PAD>']

        device = feats.device

        # note
        backpointers = []
        # Initialize the viterbi variables in log space
        # In[4]: torch.full((1, 3), -10000.)
        # Out[4]: tensor([[-10000., -10000., -10000.]])
        # size = (1,3)
        # size = (10,20)
        init_vvars = torch.full((batch_size, out_size), -10000.)
        # start = 0
        # size = (10,20)
        init_vvars[:, start_id] = 0
        # forward_var at step i holds the viterbi variables for step i-1
        # = previous
        forward_var = init_vvars.to(device)

        # swap dimension of the feats
        # [10, 176, 20] -> [176, 10, 20]
        feats = torch.transpose(feats, 0, 1)

        for feat in feats:  # [batch_size, out_size]/ [10,20]
            # list of best-tag-index
            bptrs_t = []  # holds the backpointers for this step
            # list of tensors of scores
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(out_size):  # tag' index, 0 ~ len(tags)
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                # size = (10,20)
                next_tag_var = forward_var + self.transition[next_tag].expand(batch_size, out_size)

                # index, e.g. 3/ [3,3,3,3,3,3,3,3,3,3]
                best_tag_value, best_tag_id = torch.max(next_tag_var, 1)
                # record [10]
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(best_tag_value)

            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            # size: [200] -> [20, 10] -> [10, 20]
            forward_var = torch.transpose((torch.cat(viterbivars_t)).view(out_size, batch_size), 0, 1) + feat
            # bptrs_t size: [200] -> [20, 10] -> [10, 20]
            backpointers.append(torch.transpose((torch.cat(bptrs_t)).view(out_size, batch_size), 0, 1))

        """not need for end tag"""
        # # Transition to STOP_TAG
        # # size: [10, 20]
        # terminal_var = forward_var + self.transition[end_id].expand(batch_size, out_size)
        # [10], [10]
        terminal_var = forward_var
        path_score, best_tag_id = torch.max(terminal_var, 1)

        # Follow the back pointers to decode the best path.
        # [[10], ....] -> final [20, 10]
        best_path = [best_tag_id]
        # backpointers [176, 10, 20]
        for bptrs_t in reversed(backpointers):  # [10, 20]
            # bptrs_t is a [10x20] tensor
            # best_tag_id is a [10] tensor
            temp = []
            for j in range(batch_size):
                temp.append(bptrs_t[j][best_tag_id[j]].view(1, -1))
            # size [10]
            best_tag_id = torch.cat(temp, 0).reshape(-1)
            best_path.append(best_tag_id)

        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert torch.equal(start, torch.tensor(start_id).expand(batch_size).to(device))  # Sanity check

        # size [176,10] -> [10, 176]
        best_path = torch.transpose((torch.cat(best_path)).view(max_len, batch_size), 0, 1)
        best_path = torch.fliplr(best_path)

        # remove all the pads

        # [10], [10,176]
        return path_score, best_path


def test():
    global char_to_index, model, id_to_tag, device
    while True:
        text = input('请输入:')
        # text_index = [[char_to_index[i] for i in text]]  # add [] dim this dim = 1
        text_index = [[char_to_index.get(i, char_to_index["<UNK>"]) for i in text]]
        text_index = torch.tensor(text_index, dtype=torch.int64, device=device)

        if BI_LSTM_CRF:
            prediction = model.test(text_index, [len(text)])
            prediction = [id_to_tag[i] for i in prediction]
        else:
            model.forward(text_index)  # no target/tag_index
            # get the predicted tag
            prediction = [id_to_tag[i] for i in model.prediction]

        print([f'{w}_{s}' for w, s in zip(text, prediction)])


def final_test_BiLSTM_CRF(test_dataloader):
    global char_to_index, model, id_to_tag, device, tag_to_id, id_to_char
    # evaluation
    model.eval()  # F1, acc, recall, F1 score
    # 验证时不做更新
    with torch.no_grad():
        # need to recall all of them
        all_pre_test = []
        all_tag_test = []
        all_char_test = []

        # we do it batch by batch
        for test_batch_char_index, test_batch_tag_index, batch_len, test_batch_onerad_index in test_dataloader:
            score, pre_tag = model.prediction(test_batch_char_index, test_batch_onerad_index)
            all_pre_test.extend(pre_tag[:, :-1].detach().cpu().numpy().reshape(-1).tolist())
            all_tag_test.extend(test_batch_tag_index[:, :-1].detach().cpu().numpy().reshape(-1).tolist())
            # get characters
            all_char_test.extend(test_batch_char_index[:, :-1].detach().cpu().numpy().reshape(-1).tolist())
            # temp = test_batch_char_index[:, :-1].detach().cpu().numpy().reshape(-1).tolist()
            # print(temp)

        # statistics
        length_all = len(all_tag_test)

        # remove all O s
        if REMOVE_O:  # true
            # find the index of O tag:
            O_id = tag_to_id['O']
            # find all O
            O_tag_indices = [i for i in range(length_all)
                             if all_tag_test[i] == O_id]
            # get rid of Os
            all_tag_test = [tag for i, tag in enumerate(all_tag_test)
                            if i not in O_tag_indices]
            all_pre_test = [tag for i, tag in enumerate(all_pre_test)
                            if i not in O_tag_indices]
            all_char_test = [tag for i, tag in enumerate(all_char_test)
                             if i not in O_tag_indices]
            # report
            print("原总标记数为{}，移除了{}个O标记，占比{:.2f}%".format(
                length_all,
                len(O_tag_indices),
                len(O_tag_indices) / length_all * 100
            ))
        # calculate score
        test_score = f1_score(all_tag_test, all_pre_test, average='micro')  # micro/多类别的
        # print('score')

        # prediction = [id_to_tag[i] for i in model.prediction]

        print(f'final_test: f1_score:{test_score:.3f}.)')

        if SHOW_REPORT:  # true
            # show input character
            words = [id_to_char[i] for i in all_char_test]

            # show output prediction
            prediction = [id_to_tag[i] for i in all_pre_test]

            # show output truth
            ground_truth = [id_to_tag[i] for i in all_tag_test]

            with open('Report/RawReport.txt', 'w', encoding='utf-8') as f:
                for i in range(len(words)):
                    f.write(words[i] + '\t' + prediction[i] + '\t' + ground_truth[i] + '\n')


def final_test_BiLSTM(test_dataloader):
    global char_to_index, model, id_to_tag, device, tag_to_id
    # evaluation
    model.eval()  # F1, acc, recall, F1 score
    # 验证时不做更新
    with torch.no_grad():
        # need to recall all of them
        all_pre_test = []
        all_tag_test = []
        all_char_test = []

        # we do it batch by batch
        for test_batch_char_index, test_batch_tag_index, batch_len, test_batch_onerad_index in test_dataloader:
            # loss
            test_loss = model.forward(test_batch_char_index, test_batch_onerad_index, test_batch_tag_index)
            # score
            all_pre_test.extend(model.prediction.cpu().numpy().tolist())
            # reshape(-1): 一句话里面很多字，全部拉平
            all_tag_test.extend(test_batch_tag_index.cpu().numpy().reshape(-1).tolist())
            # get characters
            all_char_test.extend(test_batch_char_index[:, :-1].detach().cpu().numpy().reshape(-1).tolist())

        # statistics
        length_all = len(all_tag_test)

        # remove all O s
        if REMOVE_O:  # true
            # find the index of O tag:
            O_id = tag_to_id['O']
            # find all O
            O_tag_indices = [i for i in range(length_all)
                             if all_tag_test[i] == O_id]
            # get rid of Os
            all_tag_test = [tag for i, tag in enumerate(all_tag_test)
                            if i not in O_tag_indices]
            all_pre_test = [tag for i, tag in enumerate(all_pre_test)
                            if i not in O_tag_indices]
            all_char_test = [tag for i, tag in enumerate(all_char_test)
                             if i not in O_tag_indices]
            # report
            print("原总标记数为{}，移除了{}个O标记，占比{:.2f}%".format(
                length_all,
                len(O_tag_indices),
                len(O_tag_indices) / length_all * 100
            ))
        # calculate score
        test_score = f1_score(all_tag_test, all_pre_test, average='micro')  # micro/多类别的
        # print('score')

        # prediction = [id_to_tag[i] for i in model.prediction]

        print(f'final_test: f1_score:{test_score:.3f}, test_loss:{test_loss:.3f}')

        # human readable presentations
        if SHOW_REPORT:  # true
            # show input character
            words = [id_to_char[i] for i in all_char_test]

            # show output prediction
            prediction = [id_to_tag[i] for i in all_pre_test]

            # show output truth
            ground_truth = [id_to_tag[i] for i in all_tag_test]

            with open('Report/RawReport.txt', 'w', encoding='utf-8') as f:
                for i in range(len(words)):
                    f.write(words[i] + '\t' + prediction[i] + '\t' + ground_truth[i] + '\n')


def save_model(model):
    torch.save(model, 'save_model/model.pk1')  # save entire net
    torch.save(model.state_dict(), 'save_model/model_parameters.pk1')  # save dict


def load_model():
    model = torch.load('save_model/model.pk1')
    model.load_state_dict(torch.load('save_model/model_parameters.pk1'))
    model.eval()
    return model


def draw_plot(train_f1, train_loss_, dev_f1, dev_loss_, model_='LSTM', dataset_='Weibo', out_dir='output.png'):
    assert len(train_f1) == len(dev_f1)
    assert len(train_f1) == len(train_loss_)
    assert len(dev_f1) == len(dev_loss_)

    # epoch
    xdata = np.arange(0, len(train_f1))
    # data
    ydata_train_f1 = train_f1
    ydata_train_loss = train_loss_
    ydata_dev_f1 = dev_f1
    ydata_dev_loss = dev_loss_

    # set subplot
    plt.clf()
    f = plt.figure(figsize=(10,4))
    ax1 = f.add_subplot(121)
    ax2 = f.add_subplot(122)

    title = f'{model_} model on {dataset_} dataset'
    f.suptitle('\n'.join(textwrap.wrap(title, 75)))

    ax1.plot(xdata, ydata_train_f1, marker='1', linestyle='-', color='r', label='train', linewidth=1)
    ax1.plot(xdata, ydata_dev_f1, marker='1', linestyle='-', color='b', label='dev', linewidth=1)
    ax1.set(xlabel='epoch',ylabel='f1 scores')
    ax1.set_title('model f1 score')
    ax1.legend(loc='best')

    ax2.plot(xdata, ydata_train_loss, marker='1', linestyle='-', color='r', label='train', linewidth=1)
    ax2.plot(xdata, ydata_dev_loss, marker='1', linestyle='-', color='b', label='dev', linewidth=1)
    ax2.set(xlabel='epoch',ylabel='loss')
    ax2.set_title('model loss')
    ax2.legend(loc='best')

    plt.savefig(out_dir)
    plt.show()


def train_search(config, checkpoint_dir=None):
    # config =
    # embedding num  # reduce 0-300
    # hidden_num  # reduce 100-300
    # lr  # 0.001
    # epoch  # 20

    # init
    class_num = len(tag_to_id)

    # training setting
    train_batch_size = 10
    dev_batch_size = 10
    test_batch_size = 1
    bi = True


    # Data Setup
    # no shuffle ordered by the len of sentence
    # need to add padding, thus use self-defined collate_fn function
    train_dataset = MyDataset(train_data, train_tag, char_to_index, tag_to_id, id_to_radical)
    train_dataloader = DataLoader(train_dataset, train_batch_size, shuffle=False,
                                  collate_fn=train_dataset.pro_batch_data)

    # evaluation
    if DEV:
        dev_dataset = MyDataset(dev_data, dev_tag, char_to_index, tag_to_id, id_to_radical)
        dev_dataloader = DataLoader(dev_dataset, dev_batch_size, shuffle=False,
                                    collate_fn=dev_dataset.pro_batch_data)

    # test data
    test_dataset = MyDataset(test_data, test_tag, char_to_index, tag_to_id, id_to_radical)
    test_dataloader = DataLoader(test_dataset, test_batch_size, shuffle=False,
                                 collate_fn=test_dataset.pro_batch_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # SETTING
    if BI_LSTM_CRF:
        model = LSTM_CRF_Model(char_num, config["embedding_num"], total_rad_ids, config["hidden_num"], class_num, bi)
        opt = torch.optim.AdamW(model.parameters(), lr=0.001)  # Adam/AdamW
    else:
        model = LSTMModel(char_num, config["embedding_num"], total_rad_ids, config["hidden_num"], class_num, bi)
        opt = torch.optim.AdamW(model.parameters(), lr=0.001)  # Adam/AdamW
    model = model.to(device)

    # draw the curve
    train_model_f1 = []
    train_model_lost = []
    dev_model_f1 = []
    dev_model_lost = []

    for e in range(25):
        # train
        model.train()

        # need to recall all for draw
        all_pre = []
        all_tag = []

        for batch_char_index, batch_tag_index, batch_len, batch_onerad_index in train_dataloader:
            # both of them already in cuda
            train_loss = model.forward(batch_char_index, batch_onerad_index, batch_tag_index)
            train_loss.backward()
            opt.step()
            opt.zero_grad()
            # for drawing
            if BI_LSTM_CRF:
                # 10, 10x176
                score, pre_tag = model.prediction(batch_char_index, batch_onerad_index)
                all_pre.extend(pre_tag[:, :-1].detach().cpu().numpy().reshape(-1).tolist())
                all_tag.extend(batch_tag_index[:, :-1].detach().cpu().numpy().reshape(-1).tolist())
            else:  # LSTM
                # score
                all_pre.extend(model.prediction.detach().cpu().numpy().tolist())
                # reshape(-1): 一句话里面很多字，全部拉平
                all_tag.extend(batch_tag_index.detach().cpu().numpy().reshape(-1).tolist())

        # calculate score
        train_score = f1_score(all_tag, all_pre, average='micro')  # micro/多类别的
        train_model_f1.append(train_score)
        train_model_lost.append(train_loss.detach().cpu())
        # print(f'train_loss:{train_loss:.3f}')
        print(f'epoch:{e}, train_f1_score:{train_score:.5f}, train_loss:{train_loss:.5f}')

        if DEV:
            # evaluation
            model.eval()  # F1, acc, recall, F1 score
            # 验证时不做更新
            with torch.no_grad():  # detach
                # need to recall all of them
                all_pre = []
                all_tag = []

                # we do it batch by batch
                for dev_batch_char_index, dev_batch_tag_index, batch_len, dev_batch_onerad_index in dev_dataloader:
                    if BI_LSTM_CRF:
                        # self-loss-added
                        dev_loss = model.forward(dev_batch_char_index, dev_batch_onerad_index, dev_batch_tag_index)
                        # using model.test
                        score, pre_tag = model.prediction(dev_batch_char_index, dev_batch_onerad_index)
                        all_pre.extend(pre_tag[:, :-1].detach().cpu().numpy().reshape(-1).tolist())
                        all_tag.extend(dev_batch_tag_index[:, :-1].detach().cpu().numpy().reshape(-1).tolist())

                    else:  # LSTM
                        # loss
                        dev_loss = model.forward(dev_batch_char_index, dev_batch_onerad_index, dev_batch_tag_index)
                        # score
                        all_pre.extend(model.prediction.detach().cpu().numpy().tolist())
                        # reshape(-1): 一句话里面很多字，全部拉平
                        all_tag.extend(dev_batch_tag_index.detach().cpu().numpy().reshape(-1).tolist())

                # calculate score
                dev_score = f1_score(all_tag, all_pre, average='micro')  # micro/多类别的
                # print('score')
                dev_model_f1.append(dev_score)
                dev_model_lost.append(dev_loss.detach().cpu())
                print(f'epoch:{e}, dev_f1_score:{dev_score:.5f}, dev_loss:{dev_loss:.5f}')


    # Send the current training result back to Tune
    tune.report(mean_accuracy=sum(dev_model_f1)/len(dev_model_f1))
    # TODO save more
    # Graph


if __name__ == "__main__":
    # pre-setting
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # data-load in
    train_data, train_tag, char_to_index, tag_to_id = build_corpus('train', make_vocab=True,
                                                                   data_dir=f'Dataset/{DATASET}')
    if DEV:
        dev_data, dev_tag = build_corpus('dev', make_vocab=False, data_dir=f'Dataset/{DATASET}')
    test_data, test_tag = build_corpus('test', make_vocab=False, data_dir=f'Dataset/{DATASET}')

    # index -> tag
    id_to_tag = [i for i in tag_to_id]
    # index -> char
    id_to_char = [i for i in char_to_index]

    # total # of char & tags
    char_num = len(char_to_index)
    class_num = len(tag_to_id)

    # load-in the radicals
    if One_Radical:
        id_to_radical = build_one_radical(data_dir='Radical/Unihan_IRGSources.txt')
        total_rad_ids = 215
    elif Three_Radicals:
        id_to_radical, total_rad_ids = build_ids(data_dir='Radical/CHISEids.txt')
    else:  # create dummy
        id_to_radical, total_rad_ids = dummy_radical()

    """ place for parameter tuning """
    # search_space = {
    #     # "lr": tune.loguniform(1e-5, 1e-1),
    #     # tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
    #     "embedding_num": tune.qrandint(20, 400, 30),
    #     "hidden_num": tune.qrandint(20, 400, 30),
    #     # "epoch": tune.randint(15, 25)
    # }
    # # algo = BayesOptSearch(utility_kwargs={
    # #     "kind": "ucb",
    # #     "kappa": 2.5,
    # #     "xi": 0.0
    # # })
    # # algo = ConcurrencyLimiter(algo, max_concurrent=4)
    # analysis = tune.run(
    #     train_search,
    #     num_samples=20,
    #     scheduler=ASHAScheduler(metric="mean_accuracy", mode="max"),
    #     search_alg=tune.suggest.BasicVariantGenerator(),
    #     config=search_space,
    #     resources_per_trial={'gpu': 1}
    # )
    # print("Best hyperparameters found were: ", analysis.best_config)
    # dfs = analysis.trial_dataframes
    # [d.mean_accuracy.plot() for d in dfs.values()]
    #
    # exit()
    """ place for parameter tuning """

    # training setting
    epoch = 20
    train_batch_size = 10
    dev_batch_size = 10
    test_batch_size = 1
    # reduce 0-300
    embedding_num = 200
    ## reduce 100-300
    hidden_num = 200  # one direction ; bi-drectional = 2 * hidden
    bi = True
    # both direction
    lr = 0.001


    # get dataset
    # no shuffle ordered by the len of sentence
    # need to add padding, thus use self-defined collate_fn function
    train_dataset = MyDataset(train_data, train_tag, char_to_index, tag_to_id, id_to_radical)
    train_dataloader = DataLoader(train_dataset, train_batch_size, shuffle=False,
                                  collate_fn=train_dataset.pro_batch_data)

    # evaluation
    if DEV:
        dev_dataset = MyDataset(dev_data, dev_tag, char_to_index, tag_to_id, id_to_radical)
        dev_dataloader = DataLoader(dev_dataset, dev_batch_size, shuffle=False,
                                    collate_fn=dev_dataset.pro_batch_data)

    # test data
    test_dataset = MyDataset(test_data, test_tag, char_to_index, tag_to_id, id_to_radical)
    test_dataloader = DataLoader(test_dataset, test_batch_size, shuffle=False,
                                 collate_fn=test_dataset.pro_batch_data)

    # SETTING
    if BI_LSTM_CRF:
        model = LSTM_CRF_Model(char_num, embedding_num, total_rad_ids, hidden_num, class_num, bi)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)  # Adam/AdamW
    else:
        model = LSTMModel(char_num, embedding_num, total_rad_ids, hidden_num, class_num, bi)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)  # Adam/AdamW
    model = model.to(device)

    # draw the curve
    train_model_f1 = []
    train_model_lost = []
    dev_model_f1 = []
    dev_model_lost = []

    # start training
    for e in range(epoch):
        # train
        model.train()

        # need to recall all for draw
        all_pre = []
        all_tag = []

        for batch_char_index, batch_tag_index, batch_len, batch_onerad_index in train_dataloader:
            # both of them already in cuda
            train_loss = model.forward(batch_char_index, batch_onerad_index, batch_tag_index)
            train_loss.backward()
            opt.step()
            opt.zero_grad()
            # for drawing
            if BI_LSTM_CRF:
                # 10, 10x176
                score, pre_tag = model.prediction(batch_char_index, batch_onerad_index)
                all_pre.extend(pre_tag[:, :-1].detach().cpu().numpy().reshape(-1).tolist())
                all_tag.extend(batch_tag_index[:, :-1].detach().cpu().numpy().reshape(-1).tolist())
            else:  # LSTM
                # score
                all_pre.extend(model.prediction.detach().cpu().numpy().tolist())
                # reshape(-1): 一句话里面很多字，全部拉平
                all_tag.extend(batch_tag_index.detach().cpu().numpy().reshape(-1).tolist())

        # calculate score
        train_score = f1_score(all_tag, all_pre, average='micro')  # micro/多类别的
        train_model_f1.append(train_score)
        train_model_lost.append(train_loss.detach().cpu())
        # print(f'train_loss:{train_loss:.3f}')
        print(f'epoch:{e}, train_f1_score:{train_score:.5f}, train_loss:{train_loss:.5f}')

        if DEV:
            # evaluation
            model.eval()  # F1, acc, recall, F1 score
            # 验证时不做更新
            with torch.no_grad():  # detach
                # need to recall all of them
                all_pre = []
                all_tag = []

                # we do it batch by batch
                for dev_batch_char_index, dev_batch_tag_index, batch_len, dev_batch_onerad_index in dev_dataloader:
                    if BI_LSTM_CRF:
                        # self-loss-added
                        dev_loss = model.forward(dev_batch_char_index, dev_batch_onerad_index, dev_batch_tag_index)
                        # using model.test
                        score, pre_tag = model.prediction(dev_batch_char_index, dev_batch_onerad_index)
                        all_pre.extend(pre_tag[:, :-1].detach().cpu().numpy().reshape(-1).tolist())
                        all_tag.extend(dev_batch_tag_index[:, :-1].detach().cpu().numpy().reshape(-1).tolist())

                    else:  # LSTM
                        # loss
                        dev_loss = model.forward(dev_batch_char_index, dev_batch_onerad_index, dev_batch_tag_index)
                        # score
                        all_pre.extend(model.prediction.detach().cpu().numpy().tolist())
                        # reshape(-1): 一句话里面很多字，全部拉平
                        all_tag.extend(dev_batch_tag_index.detach().cpu().numpy().reshape(-1).tolist())

                # calculate score
                dev_score = f1_score(all_tag, all_pre, average='micro')  # micro/多类别的
                # print('score')
                dev_model_f1.append(dev_score)
                dev_model_lost.append(dev_loss.detach().cpu())
                print(f'epoch:{e}, dev_f1_score:{dev_score:.5f}, dev_loss:{dev_loss:.5f}')

    # Test the model:
    if BI_LSTM_CRF:
        final_test_BiLSTM_CRF(test_dataloader)
    else:
        final_test_BiLSTM(test_dataloader)

    # save model
    save_model(model)

    # load model
    load_model()

    # draw the plot
    if DRAW_GRAPH:
        # setting names
        if BI_LSTM_CRF:
            model_name = 'BI_LSTM_CRF'
        else:
            model_name = 'BI_LSTM'

        if One_Radical:
            radical_state = 'One_radical'
        elif Three_Radicals:
            radical_state = 'Three_radicals'
        else:
            radical_state = 'No_radicals'
        draw_plot(train_model_f1, train_model_lost, dev_model_f1, dev_model_lost, model_='LSTM', dataset_=DATASET, out_dir=f'Figure/{model_name}-{DATASET}-{radical_state}.png')

    # manual input
    test()