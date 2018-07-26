import torch as T 
import torch.nn as nn
from torch.autograd import Variable

import foreign.skipthought.skipthoughts as skipthought

class SkipThought:
    def __init__(self, dataset, config):
        self.dataset = dataset 

        vocab = dataset.word2idx.keys()

        self.uniskip = skipthought.UniSkip(config.paths["skip_weights"], vocab)
        self.biskip  = skipthought.BiSkip(config.paths["skip_weights"],  vocab)
        
        self.cos     = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.pdist   = nn.PairwiseDistance(p=2)

    def __call__(self, caption, query, similarity="cos"):
        c_encode = self.encode(caption)
        q_encode = self.encode(query)

        maxlen  = max(len(c_encode), len(q_encode))

        query   = [0] * maxlen
        caption = [0] * maxlen

        query[:len(q_encode)]   = q_encode
        caption[:len(c_encode)] = c_encode

        combine = Variable(T.LongTensor([caption, query]))
        uni_seq2vec = self.uniskip(combine)
        bi_seq2vec  = self.biskip(combine)
        
        combine_seq2vec = T.cat((uni_seq2vec, bi_seq2vec), 1)

        if similarity == "cos":
            dist = self.cos(combine_seq2vec[0], combine_seq2vec[1]).data[0] 
        else:
            dist = self.pdist(combine_seq2vec[0].unsqueeze(0), combine_seq2vec[1].unsqueeze(0)).data[0][0]
            dist = 1. / (1+dist)

        return dist

    def encode(self, sentence):
        return [self.dataset.word2idx.get(s, 3) for s in sentence.split(" ")]