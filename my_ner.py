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
    def __init__(self, data, tag, char2id, tag2id):
        # char to index
        self.data = data
        self.tag = tag
        self.char2id = char2id
        self.tag2id = tag2id

    def __getitem__(self,index):
        # get one sentence
        sentence = self.data[index]
        sentence_tag = self.tag[index]

        # get each char's/tag's index in each sentence
        char_index = [self.char2id[i] for i in sentence]
        tag_index = [self.tag2id[i] for i in sentence_tag]

        return char_index, tag_index

    def __len__(self):
        assert len(self.data) == len(self.tag)

        return len(self.tag)

    # function for making padding
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

        return torch.tensor(sentences, dtype=torch.int64, device=device), torch.tensor(sentences_tag, dtype=torch.int64, device=device)


class MyModel(nn.Module):
    def __init__(self, char_num, embedding_num, hidden_num, class_num, bi=True):
        super().__init__()
        self.embedding = nn.Embedding(char_num, embedding_num)
        self.lstm = nn.LSTM(embedding_num, hidden_num, num_layers=1, batch_first=True, bidirectional=bi)
        if bi:
            self.classifier = nn.Linear(hidden_num * 2, class_num)
        else:
            self.classifier = nn.Linear(hidden_num, class_num)
        self.cross_loss = nn.CrossEntropyLoss()  # it contains softmax inside

    def forward(self, batch_char_index, batch_tag_index=None):
        embedding = self.embedding(batch_char_index)
        out, hidden = self.lstm(embedding)
        prediction = self.classifier(out)
        self.prediction = torch.argmax(prediction, dim=-1).reshape(-1)

        if batch_tag_index is not None:
            # change shape to [batch_size * char_num] * class_num & [batch_size * char_num]
            loss = self.cross_loss(prediction.reshape(-1, prediction.shape[-1]), batch_tag_index.reshape(-1))
            return loss




def test():
    global char_to_index, model, id_to_tag, device
    while True:
        text = input('请输入:')
        text_index = [[char_to_index[i] for i in text]] # add dim this dim = 1
        text_index.torch.tensor(text_index, dtype=torch.int64, device=device)
        model.forward(text_index)
        prediction = [id_to_tag[i] for i in model.prediction]

        print([f'{w}_{s}' for w, s in zip(text, prediction)])


if __name__ == "__main__":
    # pre-setting
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # data-load in
    train_data, train_tag, char_to_index, tag_to_id = build_corpus('train', make_vocab=True, data_dir='Dataset/weiboNER')
    dev_data, dev_tag = build_corpus('dev', make_vocab=False, data_dir='Dataset/weiboNER')
    id_to_tag = [i for i in tag_to_id]

    # total # of char & tags
    char_num = len(char_to_index)
    class_num = len(tag_to_id)

    # training setting
    epoch = 5
    train_batch_size = 50
    dev_batch_size = 100
    embedding_num = 100
    hidden_num = 128
    bi = True
    lr = 0.001

    # get dataset
    # no shuffle ordered by the len of sentence
    # need to add padding, thus use self-defined collate_fn function
    train_dataset = MyDataset(train_data, train_tag, char_to_index, tag_to_id)
    train_dataloader = DataLoader(train_dataset, train_batch_size, shuffle=False, collate_fn=train_dataset.pro_batch_data)

    dev_dataset = MyDataset(dev_data, dev_tag, char_to_index, tag_to_id)
    dev_dataloader = DataLoader(dev_dataset, dev_batch_size, shuffle=False, collate_fn=train_dataset.pro_batch_data)

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
            all_pre = []
            all_tag = []
            for dev_batch_char_index, dev_batch_tag_index in dev_dataloader:
                # loss
                dev_loss = model.forward(dev_batch_char_index, dev_batch_tag_index)
                # score
                all_pre.extend(model.prediction.cpu().numpy().tolist())
                all_tag.extend(dev_batch_tag_index.cpu().numpy().reshape(-1).tolist())
                score = f1_score(all_tag, all_pre, average='micro')
                # print('score')
            print(f'epoch:{e}, f1_score:{f1_score:.3f}, dev_loss:{dev_loss:.3f}')

    test()

