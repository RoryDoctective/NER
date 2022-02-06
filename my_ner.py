# D:\download\anaconda\envs\ner\python.exe
# -*- coding:utf-8 -*-
# time: 2021/11/25

# the codes are learnt from https://github.com/shouxieai/nlp-bilstm_crf-ner

import os
from itertools import zip_longest
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from operator import itemgetter
import numpy as np

# global:
REMOVE_O = False
SHOW_REPORT = True

BI_LSTM_CRF = True

One_Radical = False
Three_Radicals = True


# https://github.com/luopeixiang/named_entity_recognition/blob/master/data.py
def build_corpus(split, make_vocab=True, data_dir='Dataset/weiboNER'):
    """Read in data"""

    assert split in ['train', 'dev', 'test']

    char_lists = []
    tag_lists = []

    with open(os.path.join(data_dir, 'weiboNER_2nd_conll.' + split), 'r', encoding='utf-8') as f:
        char_list = []
        tag_list = []
        for line in f:
            if line != '\n':
                word, tag = line.strip('\n').split()
                char_list.append(word[0])
                tag_list.append(tag)
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


def build_ids(data_dir='Radical/CHISEids.txt'):
    """
        Read in data
        id_to_many_radicals -> [id, ’⬚十日‘]
        'U+0080'*# = padding
        'itself'*3 = not chinese
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
        if Three_Radicals:
            for i in char_index:
                temp = []
                for j in range(3):
                    temp.append(self.id2rad[i][j])
                radical_index.append(temp)

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
                   [torch.tensor(Temp[:,:,0], dtype=torch.int64, device=device),
                    torch.tensor(Temp[:,:,1], dtype=torch.int64, device=device),
                    torch.tensor(Temp[:,:,2], dtype=torch.int64, device=device)]


# model = LSTMModel(char_num, embedding_num, hidden_num, class_num, bi)
class LSTMModel(nn.Module):
    # 多少个不重复的汉字， 多少个embedding，LSTM隐藏大小， 分类类别， 双向
    def __init__(self, char_num, embedding_num, total_rad_ids, hidden_num, class_num, bi=True):
        super().__init__()
        self.embedding = nn.Embedding(char_num, embedding_num)
        if One_Radical:
            self.one_radical_embedding = nn.Embedding(total_rad_ids, 50)
            # 一层， batch在前面
            self.lstm = nn.LSTM(embedding_num + 50, hidden_num, num_layers=1, batch_first=True, bidirectional=bi)
        elif Three_Radicals:
            self.one_radical_embedding = nn.Embedding(total_rad_ids, 100)
            self.lstm = nn.LSTM(embedding_num + 100*3, hidden_num, num_layers=1, batch_first=True, bidirectional=bi)
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
        # 每一个汉字 + 一个embedding长度 = embedding
        self.embedding = nn.Embedding(char_num, embedding_num)

        if One_Radical:
            self.one_radical_embedding = nn.Embedding(total_rad_ids, 50)
            # 一层， batch在前面
            self.lstm = nn.LSTM(embedding_num + 50, hidden_num, num_layers=1, batch_first=True, bidirectional=bi)
        elif Three_Radicals:
            self.one_radical_embedding = nn.Embedding(total_rad_ids, 100)
            self.lstm = nn.LSTM(embedding_num + 100*3, hidden_num, num_layers=1, batch_first=True, bidirectional=bi)
        else:  # no radical
            # 一层， batch在前面
            self.lstm = nn.LSTM(embedding_num, hidden_num, num_layers=1, batch_first=True, bidirectional=bi)

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
        self.transition = nn.Parameter(torch.ones(class_num, class_num) * 1 / class_num)

        # This need to be self_defined.
        self.loss_fun = self.cal_lstm_crf_loss

    # 完全看不懂
    def cal_lstm_crf_loss(self, crf_scores, targets):
        # 该损失函数的计算可以参考: https: // arxiv.org / pdf / 1603.01360.pdf
        global tag_to_id
        pad_id = tag_to_id.get('<PAD>')
        start_id = tag_to_id.get('<START>')
        end_id = tag_to_id.get('<END>')

        device = crf_scores.device

        # targets:[B, L] crf_scores:[B, L, T, T]
        batch_size, max_len = targets.size()
        target_size = len(tag_to_id)

        # mask = 1 - ((targets == pad_id) + (targets == end_id))  # [B, L]
        mask = (targets != pad_id)
        lengths = mask.sum(dim=1)
        targets = self.indexed(targets, target_size, start_id)

        # # 计算Golden scores方法１
        # import pdb
        # pdb.set_trace()
        targets = targets.masked_select(mask)  # [real_L]

        flatten_scores = crf_scores.masked_select(mask.view(batch_size, max_len, 1, 1).expand_as(crf_scores)).view(-1,
                                                                                                                   target_size * target_size).contiguous()

        golden_scores = flatten_scores.gather(dim=1, index=targets.unsqueeze(1)).sum()

        # 计算golden_scores方法２：利用pack_padded_sequence函数
        # targets[targets == end_id] = pad_id
        # scores_at_targets = torch.gather(
        #     crf_scores.view(batch_size, max_len, -1), 2, targets.unsqueeze(2)).squeeze(2)
        # scores_at_targets, _ = pack_padded_sequence(
        #     scores_at_targets, lengths-1, batch_first=True
        # )
        # golden_scores = scores_at_targets.sum()

        # 计算all path scores
        # scores_upto_t[i, j]表示第i个句子的第t个词被标注为j标记的所有t时刻事前的所有子路径的分数之和
        scores_upto_t = torch.zeros(batch_size, target_size).to(device)
        for t in range(max_len):
            # 当前时刻 有效的batch_size（因为有些序列比较短)
            batch_size_t = (lengths > t).sum().item()
            if t == 0:
                scores_upto_t[:batch_size_t] = crf_scores[:batch_size_t, t, start_id, :]
            else:
                # We add scores at current timestep to scores accumulated up to previous
                # timestep, and log-sum-exp Remember, the cur_tag of the previous
                # timestep is the prev_tag of this timestep
                # So, broadcast prev. timestep's cur_tag scores
                # along cur. timestep's cur_tag dimension
                scores_upto_t[:batch_size_t] = torch.logsumexp(
                    crf_scores[:batch_size_t, t, :, :] +
                    scores_upto_t[:batch_size_t].unsqueeze(2),
                    dim=1
                )
        all_path_scores = scores_upto_t[:, end_id].sum()

        # 训练大约两个epoch loss变成负数，从数学的角度上来说，loss = -logP
        loss = (all_path_scores - golden_scores) / batch_size
        return loss

    # 完全看不懂
    def indexed(self, targets, tagset_size, start_id):
        """将targets中的数转化为在[T*T]大小序列中的索引,T是标注的种类"""
        batch_size, max_len = targets.size()
        for col in range(max_len - 1, 0, -1):
            targets[:, col] += (targets[:, col - 1] * tagset_size)
        targets[:, 0] += (start_id * tagset_size)
        return targets

    def forward(self, batch_data, batch_onerad, batch_tag=None):
        embedding = self.embedding(batch_data)
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
        out, _ = self.lstm(embedding)

        emission = self.classifier(out)
        batch_size, max_len, out_size = emission.size()

        crf_scores = emission.unsqueeze(2).expand(-1, -1, out_size, -1) + self.transition

        if batch_tag is not None:
            loss = self.cal_lstm_crf_loss(crf_scores, batch_tag)
            return loss
        else:
            return crf_scores

    def test(self, test_sents_tensor, test_onerad_tensor, lengths):
        """使用维特比算法进行解码"""
        global tag_to_id
        start_id = tag_to_id['<START>']
        end_id = tag_to_id['<END>']
        pad = tag_to_id['<PAD>']
        tagset_size = len(tag_to_id)

        crf_scores = self.forward(test_sents_tensor, test_onerad_tensor)
        device = crf_scores.device
        # B:batch_size, L:max_len, T:target set size
        B, L, T, _ = crf_scores.size()
        # viterbi[i, j, k]表示第i个句子，第j个字对应第k个标记的最大分数
        viterbi = torch.zeros(B, L, T).to(device)
        # backpointer[i, j, k]表示第i个句子，第j个字对应第k个标记时前一个标记的id，用于回溯
        backpointer = (torch.zeros(B, L, T).long() * end_id).to(device)
        lengths = torch.LongTensor(lengths).to(device)
        # 向前递推
        for step in range(L):
            batch_size_t = (lengths > step).sum().item()
            if step == 0:
                # 第一个字它的前一个标记只能是start_id
                viterbi[:batch_size_t, step,
                :] = crf_scores[: batch_size_t, step, start_id, :]
                backpointer[: batch_size_t, step, :] = start_id
            else:
                max_scores, prev_tags = torch.max(
                    viterbi[:batch_size_t, step - 1, :].unsqueeze(2) +
                    crf_scores[:batch_size_t, step, :, :],  # [B, T, T]
                    dim=1
                )
                viterbi[:batch_size_t, step, :] = max_scores
                backpointer[:batch_size_t, step, :] = prev_tags

        # 在回溯的时候我们只需要用到backpointer矩阵
        backpointer = backpointer.view(B, -1)  # [B, L * T]
        tagids = []  # 存放结果
        tags_t = None
        for step in range(L - 1, 0, -1):
            batch_size_t = (lengths > step).sum().item()
            if step == L - 1:
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += end_id
            else:
                prev_batch_size_t = len(tags_t)

                new_in_batch = torch.LongTensor(
                    [end_id] * (batch_size_t - prev_batch_size_t)).to(device)
                offset = torch.cat(
                    [tags_t, new_in_batch],
                    dim=0
                )  # 这个offset实际上就是前一时刻的
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += offset.long()

            try:
                tags_t = backpointer[:batch_size_t].gather(
                    dim=1, index=index.unsqueeze(1).long())
            except RuntimeError:
                import pdb
                pdb.set_trace()
            tags_t = tags_t.squeeze(1)
            tagids.append(tags_t.tolist())

        tagids = list(zip_longest(*reversed(tagids), fillvalue=pad))
        tagids = torch.Tensor(tagids).long()

        # 返回解码的结果
        return tagids.reshape(-1)


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
            pre_tag = model.test(test_batch_char_index, test_batch_onerad_index, batch_len)
            all_pre_test.extend(pre_tag.detach().cpu().numpy().tolist())
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

            with open('Report/test_result.txt', 'w', encoding='utf-8') as f:
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

            with open('Report/test_result.txt', 'w', encoding='utf-8') as f:
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


if __name__ == "__main__":
    # pre-setting
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # data-load in
    train_data, train_tag, char_to_index, tag_to_id = build_corpus('train', make_vocab=True,
                                                                   data_dir='Dataset/weiboNER')
    dev_data, dev_tag = build_corpus('dev', make_vocab=False, data_dir='Dataset/weiboNER')
    test_data, test_tag = build_corpus('test', make_vocab=False, data_dir='Dataset/weiboNER')

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

    # training setting
    epoch = 10
    train_batch_size = 10
    dev_batch_size = 100
    test_batch_size = 1
    embedding_num = 200
    hidden_num = 200  # one direction ; bi-drectional = 2 * hidden
    bi = True
    lr = 0.001

    # get dataset
    # no shuffle ordered by the len of sentence
    # need to add padding, thus use self-defined collate_fn function
    train_dataset = MyDataset(train_data, train_tag, char_to_index, tag_to_id, id_to_radical)
    train_dataloader = DataLoader(train_dataset, train_batch_size, shuffle=False,
                                  collate_fn=train_dataset.pro_batch_data)

    # evaluation
    dev_dataset = MyDataset(dev_data, dev_tag, char_to_index, tag_to_id, id_to_radical)
    dev_dataloader = DataLoader(dev_dataset, dev_batch_size, shuffle=False,
                                collate_fn=dev_dataset.pro_batch_data)

    # test data
    test_dataset = MyDataset(test_data, test_tag, char_to_index, tag_to_id, id_to_radical)
    test_dataloader = DataLoader(test_dataset, test_batch_size, shuffle=False,
                                 collate_fn=test_dataset.pro_batch_data)

    # SETTING
    if BI_LSTM_CRF:
        model = LSTM_CRF_Model(char_num, embedding_num,total_rad_ids, hidden_num, class_num, bi)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)  # Adam/AdamW
    else:
        model = LSTMModel(char_num, embedding_num,total_rad_ids, hidden_num, class_num, bi)
        opt = torch.optim.Adam(model.parameters(), lr=lr)  # Adam/AdamW
    model = model.to(device)

    # start training
    for e in range(epoch):
        # train
        model.train()
        for batch_char_index, batch_tag_index, batch_len, batch_onerad_index in train_dataloader:
            # both of them already in cuda
            train_loss = model.forward(batch_char_index, batch_onerad_index, batch_tag_index)
            train_loss.backward()
            opt.step()
            opt.zero_grad()
        print(f'train_loss:{train_loss:.3f}')

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
                    # using model.test
                    pre_tag = model.test(dev_batch_char_index, dev_batch_onerad_index, batch_len)
                    all_pre.extend(pre_tag.detach().cpu().numpy().tolist())
                    all_tag.extend(dev_batch_tag_index[:, :-1].detach().cpu().numpy().reshape(-1).tolist())

                    # self-added
                    dev_loss = 0

                else:  # LSTM
                    # loss
                    dev_loss = model.forward(dev_batch_char_index, dev_batch_onerad_index, dev_batch_tag_index)
                    # score
                    all_pre.extend(model.prediction.detach().cpu().numpy().tolist())
                    # reshape(-1): 一句话里面很多字，全部拉平
                    all_tag.extend(dev_batch_tag_index.detach().cpu().numpy().reshape(-1).tolist())

            # calculate score
            score = f1_score(all_tag, all_pre, average='micro')  # micro/多类别的
            # print('score')
            print(f'epoch:{e}, f1_score:{score:.3f}, dev_loss:{dev_loss:.3f}')

    # Test the model:
    if BI_LSTM_CRF:
        final_test_BiLSTM_CRF(test_dataloader)
    else:
        final_test_BiLSTM(test_dataloader)

    # save model
    save_model(model)

    # load model
    load_model()

    # manual input
    test()
