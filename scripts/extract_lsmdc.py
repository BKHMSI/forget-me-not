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
from torchvision.models import resnet 
from torchvision.models import densenet

from fmn2.foreign.c3d import C3D

from torch.cuda import empty_cache as cclean

FPS_O = 30

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

preprocess_c3d = transforms.Compose([
    transforms.Resize((112, 200)),
    transforms.CenterCrop((112, 112)),
    transforms.ToTensor()
])

resnets = {
    50: resnet.resnet50,
    152: resnet.resnet152
}

def batch_c3d(model, raws, vid_id, vlen, width, stride, log):

    PART_SIZE = 2

    nsteps = np.ceil(vlen / stride)

    batch = []
    
    try:
        for i in range(nsteps):
            batch += [raws[i*stride:i*stride+width].unsqueeze(0)]
        batch = T.cat(batch, 0)
    except:
        line  = f"RAWS: {raws.shape}\n"
        line += f"STEPS: {steps.shape}\n"
        line += f"Steps: {steps[-5:]}\n"
        line += f"ID: {vid_id}\n"
        sys.exit(line)

    vid_feats = np.empty((batch.size(0), 4096))
    n_parts = batch.size(0) // PART_SIZE

    for j in range(n_parts):
        log2 = f'\r{log} [DenseNet {j/n_parts*100:3.0f}%]'

        print(f'{log2} >fwd  ', flush=True, end='')
        part_range = np.s_[j*PART_SIZE:(j+1)*PART_SIZE]
        v = batch[part_range]
        v = T.autograd.Variable(v).cuda()

        part_feats = model(v)

        vid_feats[part_range] = part_feats.data.cpu().numpy().reshape(part_feats.shape[:2])

        print(f'{log2} ....  ', flush=True, end='')
        del v
        cclean()


    del raws
    cclean()

    return vid_feats.reshape(-1)

def batch_imagenet(model, raws, vid_id, vlen, fps, stride, log):
    ## raws : T.Tensor(vlen, pic_d)
    #stride_t = stride / FPS_O
    PART_SIZE = 20
    
    #dur = vlen / fps
    #stride_f = fps * stride_t
    #nsteps   = int(np.floor(dur / stride_t))

    nsteps = vlen // stride

    steps = np.arange(nsteps)
    steps = T.LongTensor(np.int32(np.floor(steps * stride)))[:-1]
    
    try:
        raws = raws[steps]
    except:
        line  = f"RAWS: {raws.shape}\n"
        line += f"STEPS: {steps.shape}\n"
        line += f"Steps: {steps[-5:]}\n"
        line += f"ID: {vid_id}\n"
        sys.exit(line)

    vid_feats = np.empty((raws.size(0), 2048))
    n_parts = raws.size(0) // PART_SIZE

    for j in range(n_parts):
        log2 = f'\r{log} [ImageNet {j/n_parts*100:3.0f}%]'

        print(f'{log2} >gpu  ', flush=True, end='')
        part_range = np.s_[j*PART_SIZE:(j+1)*PART_SIZE]
        v = raws[part_range]
        if (v!=v).sum() > 0:
            print("ERROR! NaN before FWD")
            sys.exit()
        v = T.autograd.Variable(v).cuda()

        print(f'{log2} >fwd  ', flush=True, end='')
        part_feats = model(v)

        print(f'{log2} >cpu  ', flush=True, end='')
        vid_feats[part_range] = part_feats.data.cpu().numpy().reshape(part_feats.shape[:2])

        print(f'{log2} ....  ', flush=True, end='')
        del v
        cclean()


    del raws
    cclean()

    return vid_feats.reshape(-1)


class C3DWrapper:
    def __init__(self, c3d_model_path):
        self.c3d = C3D()    
        self.c3d.load_state_dict(T.load(c3d_model_path))
        self.c3d.cuda() 
        self.c3d.eval()

    def __call__(self, x):
        return self.c3d(x)


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

class ResNet:
    resnets = {
        50: resnet.resnet50,
        152: resnet.resnet152
    }

    def __init__(self, i):
        model = ResNet.resnets[i](pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
        model.eval()
        model.cuda()
        self.model = model
        
    def __call__(self, x):
        out = self.model(x)
        return out


def time2sec(time_str):
    return sum(x * int(t) for x, t in zip([3600, 60, 1, 0.001], time_str.split(".")))


def c3d_fwd(args, movies):
    vids_path = args.data
    out_path  = args.out

    error_log = open('errors_lsmdc3.log', 'a+')

    dataset_write_interval = 10
    c3d_db = h5.File(out_path, 'a')

    if '/features' in c3d_db:
        c3d_feats = c3d_db['features']
    else:
        c3d_feats = c3d_db.create_group('features')


    c3d_m = C3DWrapper(args.c3d)

    for movie in movies:
        vid_ids = []
        vid_files = []
        movie_path = os.path.join(vids_path, movie)
        vids_list = os.listdir(movie_path)
        num_vids  = len(vids_list)
        
        for v in vids_list:
            v_id = v[:v.rfind('.')]
            movie_id = v_id[:4]
            time_range = v_id[v_id.rfind('_')+1:]
            start = time_range[:time_range.find('-')]
            end   = time_range[time_range.find('-')+1:]
            start = time2sec(start)
            end   = time2sec(end)
            v_id  = f"{movie_id}_{start}-{end}" 
            if v_id in c3d_feats: continue
            vid_ids   += [v_id]
            vid_files += [v]

        n_existing = len(vid_files)
        start_vid  = num_vids - n_existing

        print(f'\nExtracting C3D features for {movie}: {num_vids - n_existing}/{num_vids} videos')

        log_line = ''
        errors = []

        for vi, vid_id in enumerate(vid_ids):
            vidx = vi + start_vid
            vid_file = vid_files[vi]

            try:
                reader   = iio.get_reader(os.path.join(movie_path, vid_file), '.avi')
                metadata = Bunch(reader.get_meta_data())
                vlen     = int(reader.get_length())
            except:
                print("Error!!")
                d = str((vidx, vid_id, vid_file))
                m = str(metadata)
                error_log.write(f"{d}\n{m}\n")
                error_log.flush()
                ## We're skipping this bad file (probably a .webm)
                continue

            raw = []

            for i in range(vlen):
                try:
                    r = reader.get_data(i)
                    r = Image.fromarray(r)
                    r = preprocess_c3d(r)
                    r = r.unsqueeze_(0)
                    raw.append(r)
                except:
                    errors.append(vid_id)
                    vlen = i
                    break
                
            raw = T.cat(raw, 0)

            clear_line = '\r' + (' ' * 100) + '\r'
            log_line = f'{clear_line} ({vidx+1:4}) {vid_file:18} | {vidx/num_vids*100:3.1f}% {vi+1}/{len(vid_ids)} videos'
            print(f'\r{log_line}', end='')

            vid_feats = batch_c3d(c3d_m, raw, vid_id, vlen, args.width, args.stride, log_line)

            c3d_feats[vid_id] = vid_feats
            if vidx % dataset_write_interval or num_vids-vidx <= dataset_write_interval:
                c3d_db.flush()
            cclean()


def imagenet(args, movies, flag = False):
    vids_path = args.data
    out_path  = args.out

    error_log = open('errors_lsmdc_resnet.log', 'a+')

    dataset_write_interval = 10
    model_db = h5.File(out_path, 'a')

    if '/features' in model_db:
        model_feats = model_db['features']
    else:
        model_feats = model_db.create_group('features')


    imagenet_m = DenseNet(args.densenet) if flag else ResNet(args.resnet)
    print("Using {}".format("DenseNet" if flag else "ResNet"))

    for movie in movies:
        vid_ids = []
        vid_files = []
        movie_path = os.path.join(vids_path, movie)
        vids_list = os.listdir(movie_path)
        num_vids  = len(vids_list)
        
        for v in vids_list:
            v_id = v[:v.rfind('.')]
            movie_id = v_id[:4]
            time_range = v_id[v_id.rfind('_')+1:]
            start = time_range[:time_range.find('-')]
            end   = time_range[time_range.find('-')+1:]
            start = time2sec(start)
            end   = time2sec(end)
            v_id  = f"{movie_id}_{start}-{end}" 
            if v_id in model_feats: continue
            vid_ids   += [v_id]
            vid_files += [v]

        n_existing = len(vid_files)
        start_vid  = num_vids - n_existing

        print(f'Extracting features for {movie}: {num_vids - n_existing}/{num_vids} videos')

        log_line = ''
        errors = []

        for vi, vid_id in enumerate(vid_ids):
            vidx = vi + start_vid
            vid_file = vid_files[vi]

            try:
                reader   = iio.get_reader(os.path.join(movie_path, vid_file), '.avi')
                metadata = Bunch(reader.get_meta_data())
                vlen     = int(reader.get_length())
            except:
                print("Error!!")
                d = str((vidx, vid_id, vid_file))
                m = str(metadata)
                error_log.write(f"{d}\n{m}\n")
                error_log.flush()
                ## We're skipping this bad file (probably a .webm)
                continue

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

            clear_line = '\r' + (' ' * 100) + '\r'
            log_line = f'{clear_line} ({vidx+1:4}) {vid_file:18} | {vidx/num_vids*100:3.1f}% {vi+1}/{len(vid_ids)} videos'
            print(f'\r{log_line}', end='')

            vid_feats = batch_imagenet(imagenet_m, raw, vid_id, vlen, metadata.fps, args.stride, log_line)

            model_feats[vid_id] = vid_feats
            if vidx % dataset_write_interval or num_vids-vidx <= dataset_write_interval:
                model_db.flush()
            cclean()
            
if __name__ == '__main__':
    movies = [
        "0009_Forrest_Gump",
        #"0019_Pulp_Fiction",
        #"0024_THE_LORD_OF_THE_RINGS_THE_FELLOWSHIP_OF_THE_RING",
        #"1010_TITANIC",
        #"1027_Les_Miserables",
    ]

    print('>>> Running as main script <<<')

    parser = argparse.ArgumentParser(
        description='Extract C3D, ResNet, DensNet, P3D features for a list of videos at specified stride and width'
    )

    #parser.add_argument('list', metavar='DATA_LIST_PATH', type=str)
    parser.add_argument('data', metavar='DATA_PATH', type=str)
    parser.add_argument('out', metavar='OUTPUT_H5', type=str)
    parser.add_argument('--c3d', metavar='C3D_WEIGHTS_PATH', type=str, default=None)
    parser.add_argument('--resnet', type=int, default=50)
    parser.add_argument('--densenet', type=int, default=161)
    parser.add_argument('--width', metavar='WINDOW_STEP', type=int, default=16)
    parser.add_argument('--stride', metavar='WINDOW_STRIDE', type=int, default=8)
    parser.add_argument('--cuda', '-c', type=int, default=0)
    # parser.add_argument('--nfeats', metavar='MAX_N_FEATURES', type=int)

    args = parser.parse_args()

    with T.cuda.device(args.cuda):
        imagenet(args, movies)

    print('>>> EXIT SUCCESS')

## Base data
'''
frame_num = 0
db_path = "data/AN/densenet_feats_val2.h5"
vids_path = "fmn/data/ActivityNet/videos/validation"

db_feats = h5.File(db_path, 'r')
vid_ids  = list(db_feats['features'].keys())

video_id = vid_ids[np.random.randint(len(vid_ids))]
print("Video: {}".format(video_id))

video_file = f"{video_id}.mp4"

vid_path = os.path.join(vids_path, video_file)

reader = iio.get_reader(vid_path, '.mp4')

frame = reader.get_data(frame_num)
frame = preprocess(Image.fromarray(frame))
frame = frame.unsqueeze_(0)
frame = T.autograd.Variable(frame).cuda()


## Classification layer only
densenet_model = ResNet(152)
out = densenet_model(frame)
out = out.data.cpu().numpy().reshape(out.shape[:2])
print("Shape: ", out.shape)
'''