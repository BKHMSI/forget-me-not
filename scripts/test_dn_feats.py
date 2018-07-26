import os
import io
import requests
import h5py as h5
import numpy as np
from PIL import Image
import imageio as iio

import torch.nn as nn
import torch.nn.functional as F

import torch as T
from torchvision import models, transforms
from torch.autograd import Variable

# import src.tools.utils as us
np.random.seed()

class DenseNetTest:
    def __init__(self):
        self.densenet_model = models.densenet161(pretrained=True)
        self.densenet_model.eval()
        self.densenet_model.cuda()

        self.classifier = self.densenet_model.classifier
        self.classifier.eval()

        model_head = nn.Sequential(*list(self.densenet_model.children())[:-1])
        model_head.eval()
        model_head.cuda()
        self.model_head = model_head
 
    def __call__(self, x, mode = 0):
        if mode == 0:
            return self.classifier(x)
        elif mode == 1:
            return self.densenet_model(x)
        else:
            f = self.model_head(x)
            out = F.relu(f, inplace=True)
            out = F.avg_pool2d(out, kernel_size=7, stride=1).view(f.size(0), -1)
            return self.classifier(out)

class ResNetTest:
    def __init__(self):
        self.resnetmodel = models.resnet152(pretrained=True)
        self.resnetmodel.eval()
        self.resnetmodel.cuda()

        self.classifier = nn.Sequential(list(self.resnetmodel.children())[-1])
        self.classifier.eval()
        self.classifier.cuda()
        
        model_head = nn.Sequential(*list(self.resnetmodel.children())[:-1])
        model_head.eval()
        model_head.cuda()
        self.model_head = model_head
 
    def __call__(self, x, mode = 0):
        if mode == 0:
            return self.classifier(x)
        elif mode == 1:
            return self.resnetmodel(x)
        else:
            f = self.model_head(x)
            return self.classifier(out)

LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'

class ANC_DenseNet:
    def __init__(self, vid_ids, config):
        self.vid_ids = vid_ids
        self.an_path = config.paths['AN_path']

        video_lst = sorted(open(f"{self.an_path}/list.txt").read().split('\n'))
        files_lst = os.listdir()

        ## List of strings of video range indices for feature databases
        range_lst = [x[5:-3] for x in files_lst]
        ## DB start indices
        index_lst = [int(x[:len(x)/2]) for x in range_lst]

        ## Video index in global list :> database range tag
        ranges = [range_lst[us.find_le(index_lst, video_lst.index(vid))] for vid in vid_ids]
        vid_to_range = {vid_ids[i]: ranges[i] for i in range(len(vid_ids))}


def run():
    mean=np.array([0.485, 0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # FIXME This is wrong
    # normalize = lambda x: (np.array(x) - mean)/std

    ## TODO! Check with DenseNet preprocessing (size and crop)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    preprocess_c3d = transforms.Compose([
        transforms.Resize((112, 200)),
        transforms.CenterCrop((112, 112)),
        transforms.ToTensor()
    ])

    ## Base data
    frame_num = 100
    video_id  = 'v_O-YKLVm0ciI'
    db_path = "../data/AN/keras_c3d.h5"
    vids_path = "../fmn/data/ActivityNet/videos/training"

    db_feats = h5.File(db_path, 'r')
    vid_ids  = list(db_feats.keys())
    video_id = vid_ids[np.random.randint(len(vid_ids))]
    print("Video: {}".format(video_id))

    video_file = f"{video_id}.mp4"



    vid_path = os.path.join(vids_path, video_file)

    reader = iio.get_reader(vid_path, '.mp4')

    vid_feats = db_feats[video_id]


    print("Vid Feat Shape: ", vid_feats.shape)

    vid_feats = vid_feats[frame_num//8]
    frame = reader.get_data(frame_num)
    frame = preprocess(Image.fromarray(frame))
    frame = frame.unsqueeze_(0)
    frame = Variable(frame).cuda()

    feat = T.FloatTensor(vid_feats).unsqueeze_(0)
    feat = Variable(feat).cuda()

    ## Classification layer only
    model = ResNetTest()
    fc_out1 = model(feat)
    fc_out2 = model(frame, mode=1)
    # fc_out3 = model(frame, mode=2)


    labels = {int(key):value for (key, value)
            in requests.get(LABELS_URL).json().items()}

    top1_five = fc_out1.data.cpu().numpy()[0].argsort()[::-1][:5]
    top2_five = fc_out2.data.cpu().numpy()[0].argsort()[::-1][:5]
    # top3_five = fc_out3.data.cpu().numpy()[0].argsort()[::-1][:5]

    for idx in top1_five:
        print(labels[idx])
    print('-'*20)
    for idx in top2_five:
        print(labels[idx])

    # print('-'*20)
    # for idx in top3_five:
    #     print(labels[idx])

if __name__ == "__main__":
    with T.cuda.device(1):
        run()