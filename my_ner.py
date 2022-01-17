# D:\download\anaconda\envs\ner\python.exe
# -*- coding:utf-8 -*-
# time: 2021/11/25

# the codes are learnt from https://github.com/shouxieai/nlp-bilstm_crf-ner

import os
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

# global:
REMOVE_O = True

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
                char_lists.append(char_list)
                tag_lists.append(tag_list)
                char_list = []
                tag_list = []

    # shortest sentences first, longest sentences last
    char_lists = sorted(char_lists, key=lambda x: len(x), reverse=False)
    tag_lists = sorted(tag_lists, key=lambda x: len(x), reverse=False)

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


class MyDataset(Dataset):  # Inherit the torch Dataset
    # 汉字，标签
    def __init__(self, data, tag, char2id, tag2id):
        # char to index/ 汉字变数字
        self.data = data
        self.tag = tag
        self.char2id = char2id
        self.tag2id = tag2id

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

        return char_index, tag_index

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

        for sentence, sentence_tag in batch_data:
            sentences.append(sentence)
            sentences_tag.append(sentence_tag)
            batch_lens.append(len(sentence))
        batch_max_len = max(batch_lens)

        # add padding
        sentences = [i + [self.char2id['<PAD>']] * (batch_max_len - len(i)) for i in sentences]
        sentences_tag = [i + [self.tag2id['<PAD>']] * (batch_max_len - len(i)) for i in sentences_tag]

        # return tensor
        return torch.tensor(sentences, dtype=torch.int64, device=device), torch.tensor(sentences_tag, dtype=torch.int64,
                                                                                       device=device)


class MyModel(nn.Module):
    # 多少个不重复的汉字， 多少个embedding，LSTM隐藏大小， 分类类别， 双向
    def __init__(self, char_num, embedding_num, hidden_num, class_num, bi=True):
        super().__init__()

        # 每一个汉字 + 一个embedding长度 = embedding
        self.embedding = nn.Embedding(char_num, embedding_num)
        # 一层， batch在前面
        self.lstm = nn.LSTM(embedding_num, hidden_num, num_layers=1, batch_first=True, bidirectional=bi)

        if bi:  # 双向：hidden# * 2
            self.classifier = nn.Linear(hidden_num * 2, class_num)
        else:
            self.classifier = nn.Linear(hidden_num, class_num)

        self.cross_loss = nn.CrossEntropyLoss()  # it contains softmax inside

    def forward(self, batch_char_index, batch_tag_index=None):
        # if returns [5,4,101] means 5 batches, each batch 4 char, each char 101 dim represent.
        embedding = self.embedding(batch_char_index)  # get embedding
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


def test():
    global char_to_index, model, id_to_tag, device
    while True:
        text = input('请输入:')
        text_index = [[char_to_index[i] for i in text]]  # add [] dim this dim = 1
        text_index = torch.tensor(text_index, dtype=torch.int64, device=device)
        model.forward(text_index)  # no target/tag_index
        # get the predicted tag
        prediction = [id_to_tag[i] for i in model.prediction]

        print([f'{w}_{s}' for w, s in zip(text, prediction)])


def final_test(test_dataloader):
    global char_to_index, model, id_to_tag, device, tag_to_id
    # evaluation
    model.eval()  # F1, acc, recall, F1 score
    # 验证时不做更新
    with torch.no_grad():
        # need to recall all of them
        all_pre_test = []
        all_tag_test = []

        # we do it batch by batch
        for test_batch_char_index, test_batch_tag_index in test_dataloader:
            # loss
            test_loss = model.forward(test_batch_char_index, test_batch_tag_index)
            # score
            all_pre_test.extend(model.prediction.cpu().numpy().tolist())
            # reshape(-1): 一句话里面很多字，全部拉平
            all_tag_test.extend(test_batch_tag_index.cpu().numpy().reshape(-1).tolist())

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

    # total # of char & tags
    char_num = len(char_to_index)
    class_num = len(tag_to_id)

    # training setting
    epoch = 100
    train_batch_size = 10
    dev_batch_size = 100
    test_batch_size = 1
    embedding_num = 300
    hidden_num = 128  # one direction ; bi-drectional = 2 * hidden
    bi = True
    lr = 0.001

    # get dataset
    # no shuffle ordered by the len of sentence
    # need to add padding, thus use self-defined collate_fn function
    train_dataset = MyDataset(train_data, train_tag, char_to_index, tag_to_id)
    train_dataloader = DataLoader(train_dataset, train_batch_size, shuffle=False,
                                  collate_fn=train_dataset.pro_batch_data)

    # evaluation
    dev_dataset = MyDataset(dev_data, dev_tag, char_to_index, tag_to_id)
    dev_dataloader = DataLoader(dev_dataset, dev_batch_size, shuffle=False,
                                collate_fn=dev_dataset.pro_batch_data)

    # test data
    test_dataset = MyDataset(test_data, test_tag, char_to_index, tag_to_id)
    test_dataloader = DataLoader(test_dataset, test_batch_size, shuffle=False,
                                collate_fn=test_dataset.pro_batch_data)

    # test
    model = MyModel(char_num, embedding_num, hidden_num, class_num, bi)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)

    # start training
    for e in range(epoch):
        # train
        model.train()
        for batch_char_index, batch_tag_index in train_dataloader:
            # both of them already in cuda
            train_loss = model.forward(batch_char_index, batch_tag_index)
            train_loss.backward()
            opt.step()
            opt.zero_grad()
        print(f'train_loss:{train_loss:.3f}')

        # evaluation
        model.eval()  # F1, acc, recall, F1 score
        # 验证时不做更新
        with torch.no_grad():
            # need to recall all of them
            all_pre = []
            all_tag = []

            # we do it batch by batch
            for dev_batch_char_index, dev_batch_tag_index in dev_dataloader:
                # loss
                dev_loss = model.forward(dev_batch_char_index, dev_batch_tag_index)
                # score
                all_pre.extend(model.prediction.cpu().numpy().tolist())
                # reshape(-1): 一句话里面很多字，全部拉平
                all_tag.extend(dev_batch_tag_index.cpu().numpy().reshape(-1).tolist())

            # calculate score
            score = f1_score(all_tag, all_pre, average='micro')  # micro/多类别的
            # print('score')
            print(f'epoch:{e}, f1_score:{score:.3f}, dev_loss:{dev_loss:.3f}')

    # Test the model:
    final_test(test_dataloader)
    test()
