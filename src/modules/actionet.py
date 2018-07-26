import sys
import json
import h5py
import numpy as np
from skimage.transform import resize


import torch as T
from torch.autograd import Variable

import foreign.c3d as c3d
import src.tools.utils as utils

class ANC_C3D:
    def __init__(self, config):
        self.dim_f = 500
        self.c3d_feats = h5py.File(config.paths["c3d_feats"])
        self.n_time_steps = config.caption['n_video_lstm_step']
        # self.n_time_steps = config.features['n_time_steps']
        # self.train = json.load(open(config.training['data_split']['training_list_path']))


    def __call__(self, vid_ids, intervals, durations=None, is_featstamp=False):
        c3d_feats = []
        for i, vid in enumerate(vid_ids):
            video_c3d_features = self.c3d_feats[vid]['c3d_features']
            n_feats = video_c3d_features.shape[0]
            for interval in intervals[i]:
                sf, ef = interval if is_featstamp else utils.timestamp_to_featstamp(interval, n_feats, durations[i])
                feats = np.zeros((self.n_time_steps, self.dim_f))
                if ef - sf < self.n_time_steps:
                    feats[:(ef-sf), :] = video_c3d_features[sf:ef]
                else:
                    feats[...] = video_c3d_features[sf:self.n_time_steps+sf]    
                c3d_feats += [T.FloatTensor(feats).unsqueeze(0)]
        return c3d_feats


class C3D:
    def __init__(self, config):

        self.c3d = c3d.C3D()
        self.c3d.load_state_dict(T.load(config.paths["c3d_weights"]))
        self.c3d.cuda()
        self.c3d.eval()

        self.stride = config.features["window_stride"]
        self.window_width = config.features["window_width"]
        self.feat_limit   = config.features["n_time_steps"]
    
    def __call__(self, batch):
        
        batch_feats = []
        for seg in batch:
            n_steps = seg.shape[0] // self.stride
            seg_batch = []
            for step in range(n_steps):
                start = step*self.stride
                clip = np.array(seg[start:start+self.window_width])
                if clip.shape[0] == self.window_width and step < self.feat_limit:   
                    clip = np.array([resize(frame, output_shape=(112, 200), preserve_range=True, mode='constant') for frame in clip])
                    clip = clip[:, :, 44:44+112, :]  # crop centrally
                    clip = clip.transpose(3, 0, 1, 2)  # ch, fr, h, w
                    clip = T.FloatTensor(clip).unsqueeze(0)
                    clip = Variable(clip).cuda()
                    c3d_feats = self.c3d(clip)
                    seg_batch += [c3d_feats.data.cpu()]

            batch_feats += [T.cat(seg_batch, 0)]
            
            del seg_batch
            del c3d_feats
        
        return batch_feats