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

# global:
REMOVE_O = True
BI_LSTM_CRF = True

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
                    char_lists.append(char_list+["<END>"])
                    tag_lists.append(tag_list+["<END>"])
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
                                                                                       device=device), batch_lens

# model = LSTMModel(char_num, embedding_num, hidden_num, class_num, bi)
class LSTMModel(nn.Module):
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

# model = LSTM_CRF_Model(char_num, embedding_num, hidden_num, class_num, bi, tag_to_id)
class LSTM_CRF_Model(nn.Module):
    # 多少个不重复的汉字， 多少个embedding，LSTM隐藏大小， 分类类别， 双向
    def __init__(self, char_num, embedding_num, hidden_num, class_num, bi=True):
        super().__init__()
        # self.tag_to_id = tag_to_id

        # 每一个汉字 + 一个embedding长度 = embedding
        self.embedding = nn.Embedding(char_num, embedding_num)
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
                scores_upto_t[:batch_size_t] = crf_scores[:batch_size_t,t, start_id, :]
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

    def forward(self, batch_data, batch_tag=None):
        embedding = self.embedding(batch_data)
        out,_ = self.lstm(embedding)

        emission = self.classifier(out)
        batch_size, max_len, out_size = emission.size()

        crf_scores = emission.unsqueeze(2).expand(-1, -1, out_size, -1) + self.transition

        if batch_tag is not None:
            loss = self.cal_lstm_crf_loss(crf_scores, batch_tag)
            return loss
        else:
            return crf_scores

    def test(self, test_sents_tensor, lengths):
        """使用维特比算法进行解码"""
        global tag_to_id
        start_id = tag_to_id['<START>']
        end_id = tag_to_id['<END>']
        pad = tag_to_id['<PAD>']
        tagset_size = len(tag_to_id)

        crf_scores = self.forward(test_sents_tensor)
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
                    viterbi[:batch_size_t, step-1, :].unsqueeze(2) +
                    crf_scores[:batch_size_t, step, :, :],     # [B, T, T]
                    dim=1
                )
                viterbi[:batch_size_t, step, :] = max_scores
                backpointer[:batch_size_t, step, :] = prev_tags

        # 在回溯的时候我们只需要用到backpointer矩阵
        backpointer = backpointer.view(B, -1)  # [B, L * T]
        tagids = []  # 存放结果
        tags_t = None
        for step in range(L-1, 0, -1):
            batch_size_t = (lengths > step).sum().item()
            if step == L-1:
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
        for test_batch_char_index, test_batch_tag_index, batch_len in test_dataloader:
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
    epoch = 15  # 10
    train_batch_size = 30  # 10
    dev_batch_size = 100
    test_batch_size = 1
    embedding_num = 101  # 300
    hidden_num = 107 # 128  # one direction ; bi-drectional = 2 * hidden
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

    # SETTING
    if BI_LSTM_CRF:
        model = LSTM_CRF_Model(char_num, embedding_num, hidden_num, class_num, bi)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)  # Adam/AdamW
    else:
        model = LSTMModel(char_num, embedding_num, hidden_num, class_num, bi)
        opt = torch.optim.Adam(model.parameters(), lr=lr)  # Adam/AdamW
    model = model.to(device)

    # start training
    for e in range(epoch):
        # train
        model.train()
        for batch_char_index, batch_tag_index, batch_len in train_dataloader:
            # both of them already in cuda
            train_loss = model.forward(batch_char_index, batch_tag_index)
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
            for dev_batch_char_index, dev_batch_tag_index, batch_len in dev_dataloader:
                if BI_LSTM_CRF:
                    # using model.test
                    pre_tag = model.test(dev_batch_char_index, batch_len)
                    all_pre.extend(pre_tag.detach().cpu().numpy().tolist())
                    all_tag.extend(dev_batch_tag_index[:, :-1].detach().cpu().numpy().reshape(-1).tolist())

                    # self-added
                    dev_loss = 0

                else:  # LSTM
                    # loss
                    dev_loss = model.forward(dev_batch_char_index, dev_batch_tag_index)
                    # score
                    all_pre.extend(model.prediction.detach().cpu().numpy().tolist())
                    # reshape(-1): 一句话里面很多字，全部拉平
                    all_tag.extend(dev_batch_tag_index.detach().cpu().numpy().reshape(-1).tolist())

            # calculate score
            score = f1_score(all_tag, all_pre, average='micro')  # micro/多类别的
            # print('score')
            print(f'epoch:{e}, f1_score:{score:.3f}, dev_loss:{dev_loss:.3f}')

    # Test the model:
    # final_test(test_dataloader)
    test()
