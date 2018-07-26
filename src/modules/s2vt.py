import numpy as np

import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class S2VT(nn.Module):
    def __init__(self, vocab_size, config):
        super(S2VT, self).__init__()
        self.vocab_size             = vocab_size
        self.dim_feats              = config.features["dim_feats"]
        self.n_video_lstm_step      = config.features["n_time_steps"]
        self.n_caption_lstm_step    = config.caption["n_lstm_steps"]
        self.dim_hidden             = config.caption["dim_hidden"]

        self.encode_image = nn.Linear(self.dim_feats, self.dim_hidden)

        self.Wemb = nn.Embedding(self.vocab_size, self.dim_hidden).cuda()

        self.embed_word_w = nn.Parameter(T.randn([self.dim_hidden, self.vocab_size]).cuda(), requires_grad=True)
        self.embed_word_bias = nn.Parameter(T.zeros(1, self.vocab_size).cuda(), requires_grad=True)

        self.embed_word = nn.Linear(self.dim_hidden, self.vocab_size)

        self.lstm1 = nn.LSTMCell(self.dim_hidden, self.dim_hidden)
        self.lstm2 = nn.LSTMCell(self.dim_hidden * 2, self.dim_hidden)

        self.loss = nn.CrossEntropyLoss()


    def forward(self, batch_size, features, captions, caption_mask, evaluate = False):
        self.batch_size = batch_size

        # Encoding state
        state1  = Variable(T.FloatTensor(self.batch_size, self.dim_hidden).zero_().cuda(), requires_grad=False)
        state2  = Variable(T.FloatTensor(self.batch_size, self.dim_hidden).zero_().cuda(), requires_grad=False)
        padding = Variable(T.FloatTensor(self.batch_size, self.dim_hidden).zero_().cuda(), requires_grad=False)
        padding_img = Variable(T.FloatTensor(self.batch_size, self.dim_hidden).zero_().cuda(), requires_grad=False)

        output1 = Variable(T.FloatTensor(self.batch_size, self.dim_hidden).zero_().cuda(), requires_grad=False)
        output2 = Variable(T.FloatTensor(self.batch_size, self.dim_hidden).zero_().cuda(), requires_grad=False)

        action_feat, imagenet_feat = features

        features = T.cat((action_feat, imagenet_feat), 2)
        features    = action_feat
        # video_feat  = features.view(-1, self.dim_feats)
        video_feat  = Variable(features).cuda()
        # video_feat  = self.encode_image(video_feat)
        # video_feat  = video_feat.view(batch_size, self.n_video_lstm_step, self.dim_hidden)

        video_features = video_feat.permute(1, 0, 2) # n x b x h

        total_loss = 0.
        gen_words  = [] 
        max_idx = Variable(T.LongTensor([1]*batch_size).cuda())

        ################### Encoding Stage ######################
        for i in range(self.n_video_lstm_step):
            # change image_emb indexing when data struct is finalized
            output1, state1 = self.lstm1(video_features[i, :, :], (output1, state1))
            input_concat = T.cat((padding, output1), 1)
            output2, state2 = self.lstm2(input_concat, (output2, state2))


        ################## Decoding Stage #######################
        for i in range(self.n_caption_lstm_step):
            current_embed = self.Wemb(max_idx) if evaluate else self.Wemb(captions[:, i])

            output1, state1 = self.lstm1(padding_img, (output1, state1))
            input_concat = T.cat((current_embed, output1), 1)
            output2, state2 = self.lstm2(input_concat, (output2, state2))

            logit_words = self.embed_word(output2)
            # cross_entropy = self.loss(logit_words, captions[:, i+1])
            # cross_entropy = cross_entropy * caption_mask[:, i+1]

            # total_loss = total_loss + (T.sum(cross_entropy) / self.batch_size)

            if evaluate:
                max_idx = T.max(logit_words, 1)[1]
                max_word_idx = max_idx.data.cpu().numpy()

                gen_words.append(max_word_idx)

        return total_loss, np.array(gen_words).T

    def evaluate(self, batch_size, features):
        self.batch_size = batch_size

        # Encoding state
        state1  = Variable(T.randn(self.batch_size, self.dim_hidden).cuda(), requires_grad=False)
        state2  = Variable(T.randn(self.batch_size, self.dim_hidden).cuda(), requires_grad=False)
        padding = Variable(T.FloatTensor(self.batch_size, self.dim_hidden).zero_().cuda(), requires_grad=False)
        padding_img = Variable(T.FloatTensor(self.batch_size, self.dim_hidden).zero_().cuda(), requires_grad=False)

        output1 = Variable(T.randn(self.batch_size, self.dim_hidden).cuda(), requires_grad=False)
        output2 = Variable(T.randn(self.batch_size, self.dim_hidden).cuda(), requires_grad=False)

        action_feat, imagenet_feat = features

        features = T.cat((action_feat, imagenet_feat), 2)
        video_feat  = features.view(-1, self.dim_feats)
        video_feat  = Variable(video_feat).cuda()
        video_feat  = self.encode_image(video_feat)
        video_feat  = video_feat.view(batch_size, self.n_video_lstm_step, self.dim_hidden)

        video_features = video_feat.permute(1, 0, 2) # n x b x h

        gen_words = []
        ones = Variable(T.LongTensor([1]*batch_size).cuda())
        ################### Encoding Stage ######################
        for i in range(self.n_video_lstm_step):
            # change image_emb indexing when data struct is finalized
            output1, state1 = self.lstm1(video_features[i, :, :], (output1, state1))
            input_concat = T.cat((padding, output1), 1)
            output2, state2 = self.lstm2(input_concat, (output2, state2))

        ################## Decoding Stage #######################
        for i in range(self.n_caption_lstm_step):

            current_embed = self.Wemb(ones) if i == 0 else self.Wemb(max_idx)

            output1, state1 = self.lstm1(padding_img, (output1, state1))
            input_concat = T.cat((current_embed, output1), 1)
            output2, state2 = self.lstm2(input_concat, (output2, state2))

            logit_words = self.embed_word(output2)

            max_idx = T.max(logit_words, 1)[1]
            max_word_idx = max_idx.data.cpu().numpy()

            gen_words.append(max_word_idx)

        return np.array(gen_words).T

    def evaluate_beamsearch(self, batch_size, features, beam_width = 5):
        self.batch_size = batch_size
        # Encoding state
        state1  = Variable(T.FloatTensor(self.batch_size, self.dim_hidden).zero_().cuda(), requires_grad=False)
        state2  = Variable(T.FloatTensor(self.batch_size, self.dim_hidden).zero_().cuda(), requires_grad=False)
        padding = Variable(T.FloatTensor(self.batch_size, self.dim_hidden).zero_().cuda(), requires_grad=False)
        padding_img = Variable(T.FloatTensor(self.batch_size, self.dim_hidden).zero_().cuda(), requires_grad=False)

        output1 = Variable(T.FloatTensor(self.batch_size, self.dim_hidden).zero_().cuda(), requires_grad=False)
        output2 = Variable(T.FloatTensor(self.batch_size, self.dim_hidden).zero_().cuda(), requires_grad=False)

        gen_words = []

        action_feat, imagenet_feat = features

        features = T.cat((action_feat, imagenet_feat), 2)
        features    = action_feat
        # video_feat  = features.view(-1, self.dim_feats)
        video_feat  = Variable(features).cuda()
        # video_feat  = self.encode_image(video_feat)
        # video_feat  = video_feat.view(batch_size, self.n_video_lstm_step, self.dim_hidden)

        video_features = video_feat.permute(1, 0, 2) # n x b x h
 

        one = Variable(T.LongTensor([1]).cuda())
       ################### Encoding Stage ######################
        for i in range(self.n_video_lstm_step):
            # change image_emb indexing when data struct is finalized
            output1, state1 = self.lstm1(video_features[i, :, :], (output1, state1))
            input_concat = T.cat((padding, output1), 1)
            output2, state2 = self.lstm2(input_concat, (output2, state2))


        top_props = [{"sent":[1], "prob":1, "state":[output1, state1, output2, state2]}] 
        temp_props = []
        ################## Decoding Stage #######################
        for i in range(self.n_caption_lstm_step):
            for j in range(beam_width):
                
                if i == 0 and j > 0:
                    continue

                current_prop = top_props[j]
            
                current_embed = self.Wemb(Variable(T.LongTensor([current_prop["sent"][i]]).cuda()))

                output1, state1, output2, state2 = current_prop["state"]
                output1, state1 = output1.clone(), state1.clone()
                output2, state2 = output2.clone(), state2.clone()

                output1, state1 = self.lstm1(padding_img, (output1, state1))
                input_concat = T.cat((current_embed, output1), 1)
                output2, state2 = self.lstm2(input_concat, (output2, state2))

                logit_words = self.embed_word(output2)
                top_probs, top_idxs = T.topk(logit_words, beam_width, 1)

                top_probs = top_probs.data.cpu().numpy().flatten().tolist()
                top_idxs  = top_idxs.data.cpu().numpy().flatten().tolist()

                for k in range(beam_width):
                    new_prop = {
                        "sent": list(current_prop["sent"]) + [int(top_idxs[k])], 
                        "prob": current_prop["prob"]*top_probs[k], 
                        "state": [output1, state1, output2, state2]
                    }
                    temp_props.append(new_prop)

            top_n_sent = [x["prob"] for x in temp_props]
            top_n_sent = sorted(range(len(top_n_sent)), 
                    key=lambda i:top_n_sent[i])[-beam_width:]
            
            top_props = [temp_props[i] for i in top_n_sent]
            temp_props = []

        best_prop_idx = [x["prob"] for x in top_props]
        best_prop_idx = best_prop_idx.index(max(best_prop_idx))
        
        return top_props[best_prop_idx]["sent"]
        