# D:\download\anaconda\envs\ner\python.exe
# -*- coding:utf-8 -*-
# time: 2022/2/21
# Author: Robert Guthrie

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)


def argmax(vec):
    # torch.max(torch.tensor([[1, 2, 3, 4, 5], [6, 5, 4, 3, 2]]), 1)
    # Out[14]:
    # torch.return_types.max(
    #     values=tensor([5, 6]),
    #     indices=tensor([4, 0]))

    # return the argmax as a python int
    _, idx = torch.max(vec, 1)

    # to python scalar
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    # vec of size [1, 5]
    # max_score of size []
    max_score = vec[0, argmax(vec)]
    # max_score_broadcast of size [1, 5]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    # vec - max_score
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)  # class_num

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    # total path score
    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function (all path score)
        # size = [1, len(tags)], value = -10000
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:  # feat size = [5]
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of the previous tag
                # size of [] -> [1,1] -> [1,5]
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score = the score of transitioning to next_tag from i
                # i(all i s) -> next_tag
                # size = [1, 5]
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var = the value for the edge (i -> next_tag) before we do log-sum-exp
                # size = [1,5]
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag = log-sum-exp of all the scores.
                # size = [1]
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            # size [1,5]
            forward_var = torch.cat(alphas_t).view(1, -1)
        # i (all s) -> STOP
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        # final
        alpha = log_sum_exp(terminal_var)
        return alpha

    # emission score
    # done
    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    # real path score
    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        # feats = size [11,5]
        # tags = size [11,5]
        # score = size [1]
        score = torch.zeros(1)
        # self.tag_to_ix[START_TAG] = 3
        # torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long) = tensor([3])
        # [torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags] =
        #       [tensor([3]), tensor([0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 2])]
        # tags =
        #       tensor([3, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 2])

        # add start
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])

        # do middle # i = 0-10
        for i, feat in enumerate(feats):
            # Trans: tags[i] -> tags[i + 1]
            # + Emission
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]

        # add end : tran: 2->4, last tag to end tag
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    # Find the best path and path score, given the features(emission scores).
    def _viterbi_decode(self, feats):
        # feats: 11 char of 5 features = [11,5]

        backpointers = []

        # Initialize the viterbi variables in log space
        # In[4]: torch.full((1, 3), -10000.)
        # Out[4]: tensor([[-10000., -10000., -10000.]])
        # size = (1,3)
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        # start = 0
        # [1, len(tags)]
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        # = previous
        forward_var = init_vvars

        for feat in feats:  # for each char's features: [x,x,x,x,x]
            # list of best-tag-index
            bptrs_t = []  # holds the backpointers for this step
            # list of tensors of scores
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):  # tag' index, 0 ~ len(tags)
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                # size = [1, len(tags)]
                next_tag_var = forward_var + self.transitions[next_tag]
                # index, e.g. 3
                best_tag_id = argmax(next_tag_var)
                # append 3
                bptrs_t.append(best_tag_id)
                # append value of 3 = e.g. - 1.1187
                # .view(1) to change size [] to size [1]
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            # size: [5] -> [1, 5]
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        # size: [1, 5]
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    # the loss
    def neg_log_likelihood(self, sentence, tags):
        # sentence = index_of_sentences, size 11
        # tags = index_of_tags, size 11
        # feats = emission, size = 11,5
        feats = self._get_lstm_features(sentence)
        # alpha = total path score
        forward_score = self._forward_alg(feats)
        # gold = real path socre
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)

        # e.g. tensor(2.6907), [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
        return score, tag_seq


if __name__ == "__main__":
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    EMBEDDING_DIM = 5
    HIDDEN_DIM = 4

    # Make up some training data
    training_data = [(
        "the wall street journal reported today that apple corporation made money".split(),
        "B I I I O O O B I O O".split()
    ), (
        "georgia tech is a university in georgia".split(),
        "B I O O O O B".split()
    )]

    word_to_ix = {}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # Check predictions before training
    with torch.no_grad():
        # return index ver of the sentences ver
        precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
        # return tags ver of the sentences ver
        precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
        print(model(precheck_sent))

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(
            300):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence_in, targets)
            print(loss)

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()

    # Check predictions after training
    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
        print(model(precheck_sent))
    # We got it!
