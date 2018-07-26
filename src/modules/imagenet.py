import h5py as h5
import numpy as np
from PIL import Image

import torch as T
import torch.nn as nn
from torch.autograd import Variable

import torchvision 
from torchvision import transforms 

class ANC_DenseNet:
    def __init__(self, config):
        self.dim_f = 2208
        self.feats = h5.File(config.paths["densenet_feats"], 'r')
        self.n_time_steps = config.features['n_time_steps']

    def __getitem__(self, vid_ids, interval, durations=None, is_featstamp=False):
        feats = []
        for i, vid in enumerate(vid_ids):
            video_feats = self.feats['features'][vid]
            video_feats = video_feats(-1, dim_f)
            n_feats = video_feats.shape[0]
            for interval in intervals[i]:
                sf, ef = interval if is_featstamp else utils.timestamp_to_featstamp(interval, n_feats, durations[i])
                feats = np.zeros((self.n_time_steps, self.dim_f))
                if ef - sf < self.n_time_steps:
                    feats[:(ef-sf), :] = video_feats[sf:ef]
                else:
                    feats[...] = video_feats[sf:self.n_time_steps+sf]    
                feats += [T.FloatTensor(feats).unsqueeze(0)]
        return feats


class ImageNet:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    ## TODO! Check with DenseNet preprocessing (size and crop)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    def __init__(self, model):
        # self.stride = stride
        # self.width  = width

        imagenet_model = torchvision.models.resnet50(pretrained=True)

        ## excluding classification layer
        feature_extr = nn.Sequential(*list(imagenet_model.children())[:-1])

        ## FIXME! [BADR] Uh, you're overwriting the model. The sequence of children
        ## shouldn't work as a model, I think... Code online said to assign to `.classifier`
        imagenet_model = feature_extr
        imagenet_model.eval()
        imagenet_model.cuda()
        self.imagenet_model = imagenet_model


    def __call__(self, batch):
        ## TODO! Check whether it takes (nSegs, chans, width, height)
        # batch = [seg.transpose(0, 3, 2, 1) for seg in batch]

        batch_feats = []

        for seg in batch:  
            frames = [self.preprocess(Image.fromarray(np.uint8(r))).unsqueeze(0) for r in seg]
            frames = T.cat(frames, 0)

            frames = Variable(frames).cuda()
            
            feats = self.imagenet_model(frames)
            feats = feats.view(feats.size(0), -1)
            batch_feats += [feats.data.cpu()]

            del frames
            del feats

        return batch_feats
