import os
import sys
import argparse
from time import time

import json
import h5py as h5
import numpy as np
import imageio as iio
from PIL import Image
# from skimage import 
from bunch import Bunch

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet as tvresnet
from torchvision.models import densenet

from fmn2.foreign.c3d import C3D

from torch.cuda import empty_cache as cclean

FPS_O = 30
# CUDA_NUM = 2

normalize = transforms.Normalize(
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225]
)

#mean=np.array([0.485, 0.456, 0.406])
#std=np.array([0.229, 0.224, 0.225])
#normalize = lambda x: (np.array(x) - mean)/std

## TODO! Check with DenseNet preprocessing (size and crop)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

resnets = {
    50: tvresnet.resnet50
}


def batch_densenet(model, raws, vid_id, vlen, fps, stride, log):
    ## raws : T.Tensor(vlen, pic_d)
    stride_t = stride / FPS_O
    PART_SIZE = 20
    
    dur = vlen / fps
    stride_f = fps * stride_t
    nsteps   = int(np.floor(dur / stride_t))

    steps = np.arange(nsteps)
    steps = T.LongTensor(np.int32(np.floor(steps * stride_f)))[:-3]
    
    try:
        raws = raws[steps]
    except:
        line = f"RAWS: {raws.shape}\n"
        line += f"STEPS: {steps.shape}\n"
        line += f"Steps: {steps[-5:]}\n"
        line += f"ID: {vid_id}\n"
        sys.exit(line)

    vid_feats = np.empty((raws.size(0), 2208))
    n_parts = raws.size(0) // PART_SIZE

    for j in range(n_parts):
        log2 = f'\r{log} [DenseNet {j/n_parts*100:3.0f}%]'

        print(f'{log2} >fwd  ', flush=True, end='')
        part_range = np.s_[j*PART_SIZE:(j+1)*PART_SIZE]
        v = raws[part_range]
        v = T.autograd.Variable(v).cuda()

        part_feats = model(v)

        vid_feats[part_range] = part_feats.data.cpu().numpy().reshape(part_feats.shape[:2])
        print(f'{log2} ....  ', flush=True, end='')

        del v
        cclean()


    del raws
    cclean()

    return vid_feats.reshape(-1)

class DenseNet:
    densenets = {
        161: densenet.densenet161,
        169: densenet.densenet169
    }

    def __init__(self, i):
        model = DenseNet.densenets[i](pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
        model.eval()
        model.cuda()
        self.model = model

    def __call__(self, x):
        out = self.model(x)
        out = F.relu(out, inplace=True)
        return  F.avg_pool2d(out, kernel_size=7, stride=1)


def dense_net(args):
    vids_path = args.data
    out_path  = args.out

    error_log = open('errors.log', 'a+')

    dataset_write_interval = 10
    densnet_db = h5.File(out_path, 'a')

    if '/features' in densnet_db:
        densenet_feats = densnet_db['features']
    else:
        densenet_feats = densnet_db.create_group('features')

    vid_ids = []
    vid_files = []

    vids_list = sorted([l.rstrip() for l in open(args.list)])
    num_vids  = len(vids_list)
    
    for v in vids_list:
        v_id = v[:v.rindex('.')]
        if v_id in densenet_feats: continue
        vid_ids   += [v_id]
        vid_files += [v]

    n_existing = len(vid_files)
    start_vid  = num_vids - n_existing

    print(f'Extracting features for {n_existing}/{num_vids} videos')

    densenet_m = DenseNet(args.densenet)

    log_line = ''
    errors = []

    for vi, vid_id in enumerate(vid_ids):
        vidx = vi + start_vid
        vid_file = vid_files[vi]

        clear_line = '\r' + (' ' * 100) + '\r'
        log_line = f'{clear_line} ({vidx+1:4}) {vid_file:18} | {vidx/num_vids*100:3.1f}% {vi+1}/{len(vid_ids)} videos'

        try:
            reader   = iio.get_reader(os.path.join(vids_path, vid_file), '.mp4')
            metadata = Bunch(reader.get_meta_data())
            vlen     = int(reader.get_length())
        except:
            d = str((vidx, vid_id, vid_file))
            m = str(metadata)
            error_log.write(f"{d}\n{m}\n")
            error_log.flush()
            ## We're skipping this bad file (probably a .webm)
            continue            
        
        print(f'\r{log_line} [{vlen}] >loading ', end='')

        raw = []

        for i in range(vlen):
            try:
                r = reader.get_data(i)
                r = Image.fromarray(r)
                r = preprocess(r)
                r = r.unsqueeze_(0)
                raw.append(r)
            except:
                errors.append(vid_id)
                vlen = i
                break
            
        raw = T.cat(raw, 0)

        vid_feats = batch_densenet(densenet_m, raw, vid_id, vlen, metadata.fps, args.stride, log_line)

        print(f'\r{log_line} >saving ', end='')

        densenet_feats[vid_id] = vid_feats
        if vidx % dataset_write_interval or num_vids-vidx <= dataset_write_interval:
            densnet_db.flush()
        cclean()
            
if __name__ == '__main__':
    print('>>> Running as main script <<<')

    parser = argparse.ArgumentParser(
        description='Extract C3D, ResNet, DensNet, P3D features for a list of videos at specified stride and width'
        )

    parser.add_argument('list', metavar='DATA_LIST_PATH', type=str)
    parser.add_argument('data', metavar='DATA_PATH', type=str)
    parser.add_argument('out', metavar='OUTPUT_H5', type=str)
    parser.add_argument('--c3d', metavar='C3D_WEIGHTS_PATH', type=str, default=None)
    parser.add_argument('--resnet', type=int, default=0)
    parser.add_argument('--densenet', type=int, default=161)
    parser.add_argument('--width', metavar='WINDOW_STEP', type=int, default=16)
    parser.add_argument('--stride', metavar='WINDOW_STRIDE', type=int, default=8)
    parser.add_argument('--cuda', '-c', type=int, default=0)
    # parser.add_argument('--nfeats', metavar='MAX_N_FEATURES', type=int)

    args = parser.parse_args()

    with T.cuda.device(args.cuda):
        dense_net(args)
    print('>>> EXIT SUCCESS')