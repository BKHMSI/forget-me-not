import torch as T
from torch.autograd import Variable

from bunch import Bunch
import numpy as np

from src.modules.softcap import SoftCap
from src.modules import imagenet, actionet



class DC_SoftCap:
    def __init__(self, config, vocab_size):
        self.config = config
        self.model = SoftCap(
            vocab_size,
            Bunch(config["features"]),
            Bunch(config["caption"])
        )
        self.model.cuda()
        self.results = {}

        if config.training["resume"] == 1:
            self.load_model()

    def init(self, optim):
        self.optim = optim(self.model.parameters())

    def cycle(self, xy):
        optim = self.optim
        c3d_feats    = xy[0].cuda()
        dense_feats  = xy[1].cuda()
        feats_mask   = Variable(xy[2].cuda(), requires_grad=False)
        captions     = Variable(xy[3].cuda(), requires_grad=False)
        caption_mask = Variable(xy[4].cuda(), requires_grad=False)

        batch_size = c3d_feats.shape[0]

        assert dense_feats.shape[:2] == c3d_feats.shape[:2]

        features = (c3d_feats, dense_feats)
        self.loss = 0

        optim.zero_grad()
        loss, _ = self.model(batch_size, features, feats_mask, captions, caption_mask)
        loss.backward()
        optim.step()
        self.loss = loss.data[0]

    def evaluate(self, xy, idx2word = None):
        c3d_feats    = xy[0].cuda()
        dense_feats  = xy[1].cuda()
        feats_mask   = Variable(xy[2].cuda(), requires_grad=False)
        captions     = Variable(xy[3].cuda(), requires_grad=False)
        caption_mask = Variable(xy[4].cuda(), requires_grad=False)
        vid_id       = xy[5]
        gt_times     = xy[6][0]

        batch_size = c3d_feats.shape[0]

        assert dense_feats.shape[:2] == c3d_feats.shape[:2]

        features = (c3d_feats, dense_feats)
        loss, gen_words = self.model(batch_size, features, feats_mask, captions, caption_mask, evaluate = True)
        self.loss = loss.data[0]
        # self.results[vid_id[0]] = []
        if idx2word is not None:
            for j, sent in enumerate(gen_words):
                self.results[vid_id[j]] = []
                sent = list(sent)
                eos  = sent.index(2) if 2 in sent else len(sent)
                caption   = ' '.join(idx2word[gen] for gen in sent[:eos])
                # print(f"[EVAL] {vid_id[0]} --> {caption}")
                self.results[vid_id[j]].append({
                    "sentence": caption,
                    "timestamp": gt_times
                })


    def load_model(self, model_path = None):
        model_weights_path = self.config.paths["weights_load"] if model_path == None else model_path
        self.model.load_state_dict(T.load(model_weights_path))
        print("[INFO] Weights Loaded")

    def global_lr(self):
        return self.optim.param_groups[0]['lr']

    def last_loss(self):
        return self.loss

    def get_training(self, ds):
        return ds.torch_training_feats()

    def get_evaluate(self, ds):
        return ds.torch_evaluate_feats()
