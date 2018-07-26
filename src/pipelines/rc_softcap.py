import torch as T
from torch.autograd import Variable

from bunch import Bunch
import numpy as np

from src.modules.softcap import SoftCap
from src.modules import imagenet, actionet



class RC_SoftCap:
    def __init__(self, config, vocab_size):
        self.config = config
        self.model = SoftCap(
            vocab_size,
            Bunch(config["features"]),
            Bunch(config["caption"])
        )
        self.model.cuda()
        self.c3d_m  = actionet.C3D(config)
        self.rn50_m = imagenet.ImageNet(model='resnet50')

    def init(self, optim):
        self.optim = optim(self.model.parameters())

    def cycle(self, xy):
        optim = self.optim
        raws = xy[0]
        feats_mask   = Variable(xy[1].cuda())
        captions     = Variable(xy[2].cuda())
        caption_mask = Variable(xy[3].cuda())

        batch_size = len(raws)
        intervals  = xy[4]
        vid_ids    = xy[5]
        durations  = xy[6]

        ## Frames at stride for imagenet
        stride      = self.config.features['window_stride']
        feats_limit = self.config.features['n_time_steps']
        imagenet_raws = []
        for sample in raws:
            #print(len(sample[::stride]))
            seg = sample[::stride][:feats_limit]
            imagenet_raws += [seg]

        c3d_feats    = self.c3d_m(raws)
        resnet_feats = self.rn50_m(imagenet_raws)
        resnet_feats_padded = []

        for seg in resnet_feats:
            ## FIXME! ImageNet now returns a list!?
            padded_seg = T.zeros(feats_limit, *seg.shape[1:])
            padded_seg[:seg.shape[0]] = seg

            resnet_feats_padded += [padded_seg.unsqueeze(0)]

        c3d_feats_padded = []
        for seg in c3d_feats:
            padded_seg = T.zeros(feats_limit, *seg.shape[1:])
            padded_seg[:seg.shape[0]] = seg
            c3d_feats_padded += [padded_seg.unsqueeze(0)]
        
        resnet_feats = T.cat(resnet_feats_padded, 0)
        c3d_feats    = T.cat(c3d_feats_padded, 0)

        assert resnet_feats.shape[:2] == c3d_feats.shape[:2]

        features = (c3d_feats, resnet_feats)

        optim.zero_grad()
        loss = self.model(batch_size, features, feats_mask, captions, caption_mask)
        loss.backward()
        optim.step()

        self.loss = loss.data[0]

    def global_lr(self):
        return self.optim.param_groups[0]['lr']

    def last_loss(self):
        return self.loss



        

