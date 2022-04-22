# coding: utf-8
import gensim
import torch
import torch.nn as nn



###########################################

# Load word2vec pre-train model
model = gensim.models.Word2Vec.load('.\wiki-corpus\pre_trained_char_100_iter5.txt')
weights = torch.FloatTensor(model.wv.vectors)


# Build nn.Embedding() layer
embedding = nn.Embedding.from_pretrained(weights)
embedding.requires_grad = False


# Query
query = '天'

query_id = torch.tensor(model.wv.key_to_index['天'])
# query_id = torch.tensor(1000000)

gensim_vector = torch.tensor(model.wv[query])
embedding_vector = embedding(query_id)

print(gensim_vector==embedding_vector)

print(len(model.wv.index_to_key))
#######################################

# model = gensim.models.Word2Vec.load('.\wiki-corpus\pre_trained_char_100_iter5.txt')
# more_sentences = [
#     ['Advanced', 'users', 'can', 'load', 'a', 'model',
#      'and', 'continue', 'training', 'it', 'with', 'more', 'sentences'],
# ]
# model.build_vocab(more_sentences, update=True)
# model.train(more_sentences, total_examples=model.corpus_count, epochs=model.epochs)


