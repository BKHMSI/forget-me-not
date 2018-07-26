import sys
import numpy as np
import torch as T
import torch.nn as nn
from torch.autograd import Variable


class SoftCap(nn.Module):
    def __init__(self, vocab_size, feat_config, cap_config):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim_hidden = cap_config['dim_hidden']
        self.dim_feats  = feat_config['dim_feats']
        self.n_caption_lstm_steps = cap_config['n_lstm_steps']
        self.n_video_lstm_steps   = feat_config['n_time_steps']


        # encode features to lower dimenstion using fully connected
        self.encode_image = nn.Linear(self.dim_feats, self.dim_hidden)
        self.encode_mean  = nn.Linear(self.dim_feats, self.dim_hidden)

        # attention weights
        self.embed_att_w  = nn.Parameter(T.randn([self.dim_hidden, 1]).cuda(), requires_grad=True)
        self.embed_att_Wa = nn.Parameter(T.randn([self.dim_hidden, self.dim_hidden]).cuda(), requires_grad=True)
        self.embed_att_Ua = nn.Parameter(T.randn([self.dim_hidden, self.dim_hidden]).cuda(), requires_grad=True)
        self.embed_att_ba = nn.Parameter(T.randn([self.dim_hidden]).cuda(), requires_grad=True)
        
        # word embedding
        self.Wemb = nn.Embedding(self.vocab_size, self.dim_hidden).cuda()
        self.embed_word = nn.Linear(self.dim_hidden, self.vocab_size)

        self.embed_word_W = nn.Parameter(T.randn([self.dim_hidden, self.vocab_size]).cuda(), requires_grad=True)
        self.embed_word_b = nn.Parameter(T.randn([self.vocab_size]).cuda(), requires_grad=True)

        # lstms
        self.lstm1 = nn.LSTMCell(self.dim_hidden*2, self.dim_hidden)
        self.lstm2 = nn.LSTMCell(self.dim_hidden, self.dim_hidden)
        self.lstm3 = nn.LSTMCell(self.dim_hidden, self.dim_hidden)

        # dropout
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.25)

        # softmax
        self.logit_words_softmax = nn.Softmax()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch_size, features, video_mask, captions, caption_mask, evaluate = False):
        state1  = Variable(T.FloatTensor(batch_size, self.dim_hidden).zero_().cuda(), requires_grad=False)
        state2  = Variable(T.FloatTensor(batch_size, self.dim_hidden).zero_().cuda(), requires_grad=False)
        state3  = Variable(T.FloatTensor(batch_size, self.dim_hidden).zero_().cuda(), requires_grad=False)

        output1 = Variable(T.FloatTensor(batch_size, self.dim_hidden).zero_().cuda(), requires_grad=False)
        output2 = Variable(T.FloatTensor(batch_size, self.dim_hidden).zero_().cuda(), requires_grad=False)
        output3 = Variable(T.FloatTensor(batch_size, self.dim_hidden).zero_().cuda(), requires_grad=False)
        
        action_feat, imagenet_feat = features

        features = T.cat((action_feat, imagenet_feat), 2)
        features    = action_feat
        video_feat  = features.view(-1, self.dim_feats)
        video_feat  = Variable(features).cuda()
        video_feat  = self.encode_image(video_feat)
        video_feat  = video_feat.view(batch_size, self.n_video_lstm_steps, self.dim_hidden)

        vid_emb = video_feat.permute(1, 0, 2) # n x b x h

        # n x b x h
        action_feat   = action_feat.permute(1, 0, 2)
        imagenet_feat = imagenet_feat.permute(1, 0, 2)

        action_mean    = T.mean(action_feat, 0)
        imagenet_mean  = T.mean(imagenet_feat, 0)
        feat_mean      = T.cat((action_mean, imagenet_mean), 1)
        feat_mean   = action_mean
        feat_mean   = feat_mean.view(-1, self.dim_feats)

        feat_mean   = Variable(feat_mean).cuda()
        feat_mean   = self.encode_mean(feat_mean)

        feat_mean   = feat_mean.view(batch_size, self.dim_hidden)
  
        h_prev  = feat_mean.clone()
        output1 = feat_mean.clone()

        total_loss = 0.
        gen_words  = []

        # n x h x 1
        brcst_w = self.embed_att_w.unsqueeze(0).expand(self.n_video_lstm_steps,  *self.embed_att_w.shape[:2])
        # n x h x h
        att_Ua  = self.embed_att_Ua.unsqueeze(0).expand(self.n_video_lstm_steps, *self.embed_att_Ua.shape[:2])

        # n x b x h
        image_part = T.bmm(vid_emb, att_Ua) + self.embed_att_ba
        zeros = Variable(T.FloatTensor([0]*batch_size).cuda())
        max_idx  = Variable(T.LongTensor([1]*batch_size).cuda())

        video_mask = T.t(video_mask)
    
        for i in range(self.n_caption_lstm_steps):
            ##### soft temporal attention #####
            # n x b x h
            e_i = T.tanh(T.mm(h_prev, self.embed_att_Wa) + image_part)
   
            # n x b x 1
            e_i = T.bmm(e_i, brcst_w)
            # n x b
            e_i = T.sum(e_i, 2)
            # n x b
            e_i_max = T.max(e_i, 0)[0]

            e_hat_exp = T.mul(video_mask, T.exp(e_i - e_i_max))

            # b
            denom = T.sum(e_hat_exp, dim=0)
            denom = denom + T.eq(denom, zeros).float()  + 1e-9

            # n x b x h
            e_div  = T.div(e_hat_exp, denom)
            alphas = e_div.unsqueeze(2).expand(e_div.size(0), e_div.size(1), self.dim_hidden)
            # n x b x h
            attention_list = T.mul(alphas, vid_emb)
            # b x h  
            atten = T.sum(attention_list, dim=0)

            ##### lstm #####
            current_embed   = self.Wemb(max_idx) if evaluate else self.Wemb(captions[:, i])

            embed_atten = T.cat((current_embed, atten), 1) 

            output1, state1 = self.lstm1(embed_atten, (output1, state1))
            # output1 = self.dropout1(output1)

            output2, state2 = self.lstm2(output1, (output2, state2))

            h_prev = output2.clone()
    
            logit_words  = T.mm(output2, self.embed_word_W) + self.embed_word_b        
            cross_entropy = self.loss(logit_words, captions[:, i+1])

            cross_entropy = cross_entropy * caption_mask[:, i+1]
            total_loss = total_loss + (T.sum(cross_entropy) / batch_size)

            if evaluate:
                max_idx = T.max(logit_words, 1)[1]
                max_idx_numpy = max_idx.data.cpu().numpy()
        
                gen_words.append(max_idx_numpy)
    
        return total_loss, np.array(gen_words).T


    def evaluate(self, batch_size, features, video_mask, captions = None, caption_mask = None):
        state1  = Variable(T.FloatTensor(batch_size, self.dim_hidden).zero_().cuda(), requires_grad=False)
        state2  = Variable(T.FloatTensor(batch_size, self.dim_hidden).zero_().cuda(), requires_grad=False)

        output1 = Variable(T.FloatTensor(batch_size, self.dim_hidden).zero_().cuda(), requires_grad=False)
        output2 = Variable(T.FloatTensor(batch_size, self.dim_hidden).zero_().cuda(), requires_grad=False)
        
        action_feat, imagenet_feat = features

        features = T.cat((action_feat, imagenet_feat), 2)
        video_feat  = features.view(-1, self.dim_feats)
        video_feat  = Variable(video_feat).cuda()
        video_feat  = self.encode_image(video_feat)
        video_feat = video_feat.view(batch_size, self.n_video_lstm_steps, self.dim_hidden)

        vid_emb = video_feat.permute(1, 0, 2) # n x b x h

        # n x b x h
        action_feat   = action_feat.permute(1, 0, 2)
        imagenet_feat = imagenet_feat.permute(1, 0, 2)

        action_mean    = T.mean(action_feat, 0)
        imagenet_mean  = T.mean(imagenet_feat, 0)
        feat_mean      = T.cat((action_mean, imagenet_mean), 1)
        feat_mean      = feat_mean.view(-1, self.dim_feats)

        feat_mean   = Variable(feat_mean).cuda()
        feat_mean   = self.encode_image(feat_mean)

        feat_mean   = feat_mean.view(batch_size, self.dim_hidden)
  
        h_prev  = feat_mean.clone()
        output1 = feat_mean.clone()

        gen_words  = []
        total_loss = 0.
        max_idx = None

        # n x h x 1
        brcst_w = self.embed_att_w.unsqueeze(0).expand(self.n_video_lstm_steps,  *self.embed_att_w.shape[:2])
        # n x h x h
        att_Ua  = self.embed_att_Ua.unsqueeze(0).expand(self.n_video_lstm_steps, *self.embed_att_Ua.shape[:2])

        # n x b x h
        image_part = T.bmm(vid_emb, att_Ua) + self.embed_att_ba
        zeros = Variable(T.FloatTensor([0]*batch_size).cuda())
        ones  = Variable(T.LongTensor([1]*batch_size).cuda())

        for i in range(self.n_caption_lstm_steps):
            ##### soft temporal attention #####
            # n x b x h
            e_i = T.tanh(T.mm(h_prev, self.embed_att_Wa) + image_part)
            # n x b x 1
            e_i = T.bmm(e_i, brcst_w)
            # n x b
            e_i = T.sum(e_i, 2)
            # n x b
            e_i_max = T.max(e_i, 0)[0]

            e_hat_exp = T.mul(T.t(video_mask), T.exp(e_i - e_i_max))
            # b
            denom = T.sum(e_hat_exp, 0)
            denom = denom + T.eq(denom, zeros).float()  + 1e-9

            # n x b x h
            e_div  = T.div(e_hat_exp, denom)
            alphas = e_div.unsqueeze(2).expand(e_div.size(0), e_div.size(1), self.dim_hidden)
            # n x b x h
            attention_list = T.mul(alphas, vid_emb)
            # b x h  
            atten = T.sum(attention_list, 0)

            ##### lstm #####
            current_embed = self.Wemb(ones) if i == 0 else self.Wemb(max_idx)

            embed_atten = T.cat((current_embed, atten), 1) 
            output1, state1 = self.lstm1(embed_atten, (output1, state1))
            output2, state2 = self.lstm2(output1, (output2, state2))

            h_prev = output2.clone()

            logit_words  = T.mm(output2, self.embed_word_W) + self.embed_word_b
            
            if captions is not None:
                cross_entropy = self.loss(logit_words, captions[:, i+1])

                cross_entropy = cross_entropy * caption_mask[:, i+1]
                total_loss = total_loss + (T.sum(cross_entropy) / batch_size)

            max_idx = T.max(logit_words, 1)[1]
            max_idx_numpy = max_idx.data.cpu().numpy()
       
            gen_words.append(max_idx_numpy)

        return total_loss, np.array(gen_words).T

    def evaluate_beam(self, batch_size, features, video_mask, beam_width = 5):
        state1  = Variable(T.DoubleTensor(batch_size, self.dim_hidden).zero_().cuda(), requires_grad=False)
        state2  = Variable(T.DoubleTensor(batch_size, self.dim_hidden).zero_().cuda(), requires_grad=False)

        output1 = Variable(T.DoubleTensor(batch_size, self.dim_hidden).zero_().cuda(), requires_grad=False)
        output2 = Variable(T.DoubleTensor(batch_size, self.dim_hidden).zero_().cuda(), requires_grad=False)
        
        action_feat, imagenet_feat = features

        features = T.cat((action_feat, imagenet_feat), 2)
        # print("Features: ", features.size())
        video_feat  = features.view(-1, self.dim_feats)
        #print("Vid Feats: ", video_feat.size())
        video_feat  = Variable(video_feat).cuda()
        video_feat  = self.encode_image(video_feat)
        video_feat = video_feat.view(batch_size, self.n_video_lstm_steps, self.dim_hidden)
        # print("Vid Feats: ", video_feat.size())

        vid_emb = video_feat.permute(1, 0, 2) # n x b x h

        # n x b x h
        action_feat   = action_feat.permute(1, 0, 2)
        imagenet_feat = imagenet_feat.permute(1, 0, 2)

        action_mean    = T.mean(action_feat, 0)
        imagenet_mean  = T.mean(imagenet_feat, 0)
        feat_mean      = T.cat((action_mean, imagenet_mean), 1)
        feat_mean      = feat_mean.view(-1, self.dim_feats)

        feat_mean   = Variable(feat_mean).cuda()
        feat_mean   = self.encode_image(feat_mean)

        feat_mean   = feat_mean.view(batch_size, self.dim_hidden)
  
        h_prev  = feat_mean
        output1 = feat_mean

        gen_words  = []
        total_loss = 0.
        max_idx = None

        # n x h x 1
        brcst_w = self.embed_att_w.unsqueeze(0).expand(self.n_video_lstm_steps,  *self.embed_att_w.shape[:2])
        # n x h x h
        att_Ua  = self.embed_att_Ua.unsqueeze(0).expand(self.n_video_lstm_steps, *self.embed_att_Ua.shape[:2])

        # n x b x h
        image_part = T.bmm(vid_emb, att_Ua) + self.embed_att_ba
        zeros = Variable(T.DoubleTensor([0]*batch_size).cuda())
        
        top_props  = [{"sent":[1], "prob":1, "state":[output1, state1, output2, state2, h_prev]}] * beam_width
        temp_props = []
        for i in range(self.n_caption_lstm_steps):
            for j in range(beam_width):
                current_prop = top_props[j]

                output1, state1, output2, state2, h_prev = current_prop["state"]
                h_prev = h_prev.clone()
                ##### soft temporal attention #####
                # n x b x h
                e_i = T.tanh(T.mm(h_prev, self.embed_att_Wa) + image_part)
                # n x b x 1
                e_i = T.bmm(e_i, brcst_w)
                # n x b
                e_i = T.sum(e_i, 2)
                # n x b
                e_i_max = T.max(e_i, 0)[0]

                e_hat_exp = T.mul(T.t(video_mask), T.exp(e_i - e_i_max))
                # b
                denom = T.sum(e_hat_exp, 0)
                denom = denom + T.eq(denom, zeros).double()  + 1e-9

                # n x b x h
                e_div  = T.div(e_hat_exp, denom)
                alphas = e_div.unsqueeze(2).expand(e_div.size(0), e_div.size(1), self.dim_hidden)
                # n x b x h
                attention_list = T.mul(alphas, vid_emb)
                # b x h  
                atten = T.sum(attention_list, 0)

                ##### lstm #####
                current_embed =  self.Wemb(Variable(T.LongTensor([current_prop["sent"][i]]).cuda()))

                output1, state1 = output1.clone(), state1.clone()
                output2, state2 = output2.clone(), state2.clone()

                embed_atten = T.cat((current_embed, atten), 1) 
                output1, state1 = self.lstm1(embed_atten, (output1, state1))
                output2, state2 = self.lstm2(output1, (output2, state2))

                h_prev = output2

                logit_words  = T.mm(output2, self.embed_word_W) + self.embed_word_b
                
                # cross_entropy = self.loss(logit_words, captions[:, i+1])

                # cross_entropy = cross_entropy * caption_mask[:, i+1]
                # total_loss = total_loss + (T.sum(cross_entropy) / batch_size)

                max_idx = T.max(logit_words, 1)[1]
                top_probs, top_idxs = T.topk(logit_words, beam_width, 1)

                top_probs = top_probs.data.cpu().numpy().flatten()
                top_idxs  = top_idxs.data.cpu().numpy().flatten()

                for k in range(beam_width):
                    new_prop = {
                        "sent":list(current_prop["sent"]) + [int(top_idxs[k])], 
                        "prob":current_prop["prob"]*top_probs[k], 
                        "state":[output1, state1, output2, state2, h_prev]
                    }
                    temp_props.append(new_prop)

            top_n_sent = [x["prob"] for x in temp_props]
            top_n_sent = sorted(range(len(top_n_sent)), 
                    key=lambda i:top_n_sent[i])[-beam_width:]
            
            top_props = [temp_props[i] for i in top_n_sent]
            temp_props = []

        best_prop_idx = [x["prob"] for x in top_props]
        best_prop_idx = best_prop_idx.index(max(best_prop_idx))
        
        return total_loss, [top_props[best_prop_idx]["sent"]]

    def evaluate_2(self, batch_size, features, video_mask):
        '''
            softcap without using mean values
        '''
        state1  = Variable(T.FloatTensor(batch_size, self.dim_hidden).zero_().cuda(), requires_grad=False)
        state2  = Variable(T.FloatTensor(batch_size, self.dim_hidden).zero_().cuda(), requires_grad=False)

        output1 = Variable(T.FloatTensor(batch_size, self.dim_hidden).zero_().cuda(), requires_grad=False)
        output2 = Variable(T.FloatTensor(batch_size, self.dim_hidden).zero_().cuda(), requires_grad=False)
       
        vid_emb   = features.permute(1, 0, 2)
        feat_mean = T.mean(vid_emb, 0)

        h_prev  = feat_mean
        output1 = feat_mean

        gen_words  = []
        total_loss = 0.

        # n x h x 1
        brcst_w = self.embed_att_w.unsqueeze(0).expand(self.n_video_lstm_steps,  *self.embed_att_w.shape[:2])
        # n x h x h
        att_Ua  = self.embed_att_Ua.unsqueeze(0).expand(self.n_video_lstm_steps, *self.embed_att_Ua.shape[:2])

        # n x b x h
        image_part = T.bmm(vid_emb, att_Ua) + self.embed_att_ba
        zeros = Variable(T.FloatTensor([0]*batch_size).cuda())
        ones  = Variable(T.LongTensor([1]*batch_size).cuda())

        for i in range(self.n_caption_lstm_steps):
            ##### soft temporal attention #####
            # n x b x h
            e_i = T.tanh(T.mm(h_prev, self.embed_att_Wa) + image_part)
            # n x b x 1
            e_i = T.bmm(e_i, brcst_w)
            # n x b
            e_i = T.sum(e_i, 2)
            # n x b
            e_i_max = T.max(e_i, 0)[0]

            e_hat_exp = T.mul(T.t(video_mask), T.exp(e_i - e_i_max))
            # b
            denom = T.sum(e_hat_exp, 0)
            denom = denom + T.eq(denom, zeros).float()  + 1e-9

            # n x b x h
            e_div  = T.div(e_hat_exp, denom)
            alphas = e_div.unsqueeze(2).expand(e_div.size(0), e_div.size(1), self.dim_hidden)
            # n x b x h
            attention_list = T.mul(alphas, vid_emb)
            # b x h  
            atten = T.sum(attention_list, 0)

            ##### lstm #####
            current_embed = self.Wemb(ones) if i == 0 else self.Wemb(max_idx)

            embed_atten = T.cat((current_embed, atten), 1) 
            output1, state1 = self.lstm1(embed_atten, (output1, state1))
            output2, state2 = self.lstm2(output1, (output2, state2))

            h_prev = output2

            logit_words  = T.mm(output2, self.embed_word_W) + self.embed_word_b
            
            # cross_entropy = self.loss(logit_words, captions[:, i+1])

            # cross_entropy = cross_entropy * caption_mask[:, i+1]
            # total_loss = total_loss + (T.sum(cross_entropy) / batch_size)

            max_idx = T.max(logit_words, 1)[1]
            max_idx_numpy = max_idx.data.cpu().numpy()
       
            gen_words.append(max_idx_numpy)

        return total_loss, np.array(gen_words).T
