import os
import sys

import json
from bunch import Bunch

import numpy as np
import h5py as h5

import imageio
import torch as T
from torch.utils.data import Dataset, DataLoader

import src.tools.utils as utils

class FMN_Dataset(Dataset):
    def __init__(self, dataset, vid_ids, config):
        self.ds = dataset

        self.data_path  = self.ds.data_path
        self.vid_ids    = vid_ids
        self.extensions = self.ds.extensions

        self.config = config

        self.pop_element = None

    def collate_fn_feats(self, batch):
        c3d_feats     = []
        densnet_feats = []
        feature_masks = []
        captions = []
        caption_masks = []
        vid_ids = []
        intervals = []
    
        for sample in batch:
            ## FIXME HACK! WTH is this.
            # if sample[0][0] is None: continue
            c3d_feats     += [s.unsqueeze(0) for s in sample[0]]
            densnet_feats += [s.unsqueeze(0) for s in sample[1]]
            feature_masks += [s.unsqueeze(0) for s in sample[2]]
            captions      += [s.unsqueeze(0) for s in sample[3]]
            caption_masks += [s.unsqueeze(0) for s in sample[4]]
            vid_ids       += [sample[5]]
            intervals     += [sample[6]]


        c3d_feats     = T.cat(c3d_feats, 0)
        densnet_feats = T.cat(densnet_feats, 0)
        feature_masks = T.cat(feature_masks, 0)
        captions      = T.cat(captions, 0)
        caption_masks = T.cat(caption_masks, 0)
       
        return (c3d_feats, densnet_feats, feature_masks, captions, caption_masks, vid_ids, intervals)
    
    def collate_fn_val_feats(self, batch):
        c3d_feats     = []
        densnet_feats = []
        feature_masks = []
        vid_ids       = []
    
        for sample in batch:
            ## FIXME HACK! WTH is this.
            # if sample[0][0] is None: continue
            c3d_feats     += [s.unsqueeze(0) for s in sample[0]]
            densnet_feats += [s.unsqueeze(0) for s in sample[1]]
            feature_masks += [s.unsqueeze(0) for s in sample[2]]
            vid_ids       += [sample[5]]

        c3d_feats     = T.cat(c3d_feats, 0)
        densnet_feats = T.cat(densnet_feats, 0)
        feature_masks = T.cat(feature_masks, 0)
  
        return (c3d_feats, densnet_feats, feature_masks, vid_ids)
    
    def collate_fn(self, batch):
        ## Check for collate_fn on specific dataset
        raws = []

        seq_masks = []
        captions = []
        caption_masks = []

        ints = []
        vid_ids = []
        durations = []

        for sample in batch:
            # process = lambda x: T.cat([s.unsqueeze(0) for s in sample[x]], 0)

            ## (raws, feature masks, caption indices seq, caption masks seq, interval tuples, video id)
            raws += sample[0]

            seq_masks     += [s.unsqueeze(0) for s in sample[1]]
            captions      += [s.unsqueeze(0) for s in sample[2]]
            caption_masks += [s.unsqueeze(0) for s in sample[3]]
            nsegs = len(sample[4])
            ints      += [sample[4]]
            vid_ids   += [sample[5] * nsegs]
            ## HACK! Why do we need this?
            durations += [sample[6] * nsegs]

        captions      = T.cat(captions, 0)
        seq_masks     = T.cat(seq_masks, 0)
        caption_masks = T.cat(caption_masks, 0)
        ## FIXME! List are of different length
        ## All the same except vid_ids and durations
        ## as they equal #original_videos instead of #segments
        return (raws, seq_masks, captions, caption_masks, ints, vid_ids, durations)

    def __len__(self):
        return len(self.vid_ids)

    def __getitem__(self, i):
        vid_id = self.vid_ids[i]
        intervals = self.ds.gt_intervals[vid_id]

        ## (raws, feature masks, caption indices seq, caption masks seq, interval tuples, video id) for raws_and_labels
        ## (c3d, densenet, feature masks, caption indices seq, caption masks seq) for feats_and_labels
        return self.pop_element(vid_id, intervals)

    def file_ext(self, vid_id):
        return self.extensions[vid_id]

    def feats_and_labels(self, vid_id, intervals):
        c3d_feats      = []
        densenet_feats = []

        feature_masks = []
        caption_limit = self.config['caption']['n_lstm_steps']

        duration = self.ds.duration[vid_id]
        n_time_steps = self.config.features["n_time_steps"]

        video_c3d_features     = self.ds.c3d_feats[vid_id]['c3d_features'] if self.ds.name == "anc" else self.ds.c3d_feats[vid_id]
    
        try:
            video_densenet_feature = self.ds.dense_feats[vid_id] if vid_id in self.ds.train_ids or self.ds.name == "lsmdc" else self.ds.xc_feats[vid_id]
        except:
            video_densenet_feature = np.zeros((n_time_steps, 2048))
            
        n_feats = min(video_c3d_features.shape[0], video_densenet_feature.shape[0])
        video_c3d_features = video_c3d_features[:n_feats]
        video_densenet_feature = video_densenet_feature[:n_feats]

        c3d_feats_dim = 500 if self.ds.name == "anc" else 4096

        for interval in intervals:
            mask    = np.zeros(self.config.features['n_time_steps'])
            sf, ef  = utils.timestamp_to_featstamp(interval, n_feats, duration)
            c3d_t_feats      = np.zeros((n_time_steps, c3d_feats_dim))
            densenet_t_feats = np.zeros((n_time_steps, 2048))

            if ef - sf < n_time_steps:
                n = ef - sf
                mask[:n] = 1
                c3d_t_feats[:n] = video_c3d_features[sf:ef]
                densenet_t_feats[:n] = video_densenet_feature[sf:ef]
            else:
                mask[...] = 1
                c3d_t_feats[...] = video_c3d_features[sf:n_time_steps+sf]
                densenet_t_feats[...] = video_densenet_feature[sf:n_time_steps+sf]

            #print(f"From {sf} to {ef}")
            #print("{} C3Dt: {}".format(vid_id, ((c3d_t_feats != c3d_t_feats) | (c3d_t_feats == float('inf'))).sum()))
            #print("{} Denset: {}".format(vid_id, ((densenet_t_feats != densenet_t_feats) | (densenet_t_feats == float('inf'))).sum()))

            feature_masks  += [mask] 
            c3d_feats      += [c3d_t_feats]
            densenet_feats += [densenet_t_feats]


        captions_as_indices = []
        if vid_id in self.ds.train_ids or self.ds.name == "lsmdc":
            v_captions = self.ds.captions[vid_id]
        elif vid_id in self.ds.val_ids:
            v_captions = self.ds.val_captions[vid_id]
        else:
            v_captions = [""]
 
        for cap in v_captions:
            words = cap.lower().translate(excluded_chars).split(' ')
            words = ['<bos>'] + words[:caption_limit-2] + ['<eos>']
            # if self.ds.name == "lsmdc":
            #     words = [w for w in words if w != "someone"]

            caption = [self.ds.word2idx.get(w, 3) for w in words] + [0]
            captions_as_indices.append(caption)

        caption_matrix_size = (len(captions_as_indices), caption_limit+1)
        caption_matrix = np.zeros(caption_matrix_size)
        caption_masks  = np.zeros_like(caption_matrix)

        for i, cap in enumerate(captions_as_indices):
            caption_matrix[i, :len(cap)]  = np.array(cap)
            caption_masks[i, :len(cap)-1] = 1

        return T.FloatTensor(c3d_feats), T.FloatTensor(densenet_feats), T.FloatTensor(feature_masks), T.LongTensor(caption_matrix), T.FloatTensor(caption_masks), vid_id, intervals

    def raw_and_labels(self, vid_id, intervals):
        file_ext = self.file_ext(vid_id)
        
        raws = []
        feature_masks = []
        caption_limit = self.config['caption']['n_lstm_steps']

        file_path = os.path.join(self.data_path, f'{vid_id}.{file_ext}')

        reader = imageio.get_reader(file_path, 'mp4')
        fps = reader.get_meta_data()['fps']

        duration = reader.get_meta_data()['duration']
        video = np.array(list(reader), dtype=np.uint8)

        for [s, e] in intervals:
            ## segment raws
            sf = int(s * fps)
            ef = int(np.minimum(e, duration-1)*fps)
            v = video[sf:ef]
            
            raws += [np.array(v, dtype=np.float)]

            ## segment masks
            feats_len = (ef-sf+1)//self.config.features['window_stride']
            mask = np.zeros(self.config.features['n_time_steps'])
            mask[:feats_len] = 1

            feature_masks += [mask]

        del reader

        captions_as_indices = []
        v_captions = self.ds.captions[vid_id]
        ## FIXME! Why the reassignment from `ds`? Is the duration different from
        ## that stored in the video metadata?
        # duration   = self.ds.duration[vid_id]

        for cap in v_captions:
            words = cap.lower().translate(excluded_chars).split(' ')
            ## Why did we conditionally add the end flag before?
            words = ['<bos>'] + words[:caption_limit-2] + ['<eos>']

            ## Sequence of word indices corresponding to caption words
            caption = [self.ds.word2idx.get(w, 3) for w in words] + [0]
            captions_as_indices.append(caption)

        caption_matrix_size = (len(captions_as_indices), caption_limit+1)
        caption_matrix = np.zeros(caption_matrix_size)
        caption_masks  = np.zeros_like(caption_matrix)

        for i, cap in enumerate(captions_as_indices):
            caption_matrix[i, :len(cap)] = np.array(cap)
            caption_masks[i, :len(cap)-1]  = 1

        ## (raws, feature masks, caption indices seq, caption masks seq, interval tuples, video id)
        return raws, T.FloatTensor(feature_masks), T.LongTensor(caption_matrix), T.FloatTensor(caption_masks), intervals, vid_id, duration

class LSMDC_Dataset:
    def __init__(self, paths, config):
        self.train = json.load(open(paths['data_manifest']))
        self.val = json.load(open(paths['val_data_manifest']))

        self.name = "lsmdc"

        self.config = config

        self.data_path = None
        self.extensions = None

        self.c3d_feats   = h5.File(config.paths["c3d_feats"], 'r')
        self.dense_feats = h5.File(config.paths["xception_feats"], 'r')

        self.train_ids = sorted(open(paths["training_ids"], 'r').read().split('\n'))[1:]
        self.val_ids   = open(paths["validation_ids"], 'r').read().split('\n')

        movie = ["1027"]
        

        train_ids, val_ids = [], []
        for vid in self.train_ids:
            if vid[:4] in movie:
                train_ids += [vid]

        for vid in self.val_ids:
            if vid[:4] in movie:
                val_ids += [vid]

        self.train_ids = train_ids
        self.val_ids   = val_ids

        self.duration = {}
        self.captions = {}
        self.gt_intervals = {}

        self.vocab_size = None
        self.idx2word = {}
        self.word2idx = {}


        print(f"[INFO] # of Training: {len(self.train_ids)} | Validation: {len(self.val_ids)}")

        for vid in self.train_ids:
            feat_end = min(self.dense_feats[vid].shape[0], self.c3d_feats[vid].shape[0])
            self.gt_intervals[vid] = [[0, feat_end]]
            self.duration[vid] = feat_end
            try:
                self.captions[vid] = [self.train[vid]]
            except:
                self.captions[vid] = [self.val[vid]]


        for vid in self.val_ids:
            try:
                feat_end = min(self.dense_feats[vid].shape[0], self.c3d_feats[vid].shape[0])
                self.gt_intervals[vid] = [[0, feat_end]]
                self.duration[vid] = feat_end
                try:
                    self.captions[vid] = [self.train[vid]]
                except:
                    self.captions[vid] = [self.val[vid]]
            except:
                self.val_ids.remove(vid)


        preprocess_captions(self, word_count_threshold=config['caption']['word_count_threshold'])


    def torch_training_feats(self):
        fmn = FMN_Dataset(self, self.train_ids, self.config)
        fmn.pop_element = fmn.feats_and_labels
        data_loader_args = {
            'num_workers': self.config['training']['nthreads'],
            'batch_size': self.config['training']['batch_size'],
            'shuffle': self.config['training']['shuffle'],
        }
        return DataLoader(fmn, **data_loader_args, collate_fn=fmn.collate_fn_feats)

    def torch_evaluate_feats(self):
        # val_ids = np.array(self.val_ids)[np.random.choice(len(self.val_ids), int(len(self.val_ids)*0.2))]
        fmn = FMN_Dataset(self, self.val_ids + self.train_ids, self.config)
        fmn.pop_element = fmn.feats_and_labels
        data_loader_args = {
            'num_workers': self.config['training']['nthreads'],
            'batch_size': self.config['training']['batch_size'],
            'shuffle': self.config['training']['shuffle'],
        }
        return DataLoader(fmn, **data_loader_args, collate_fn=fmn.collate_fn_feats)

class ANC_Dataset:
    def __init__(self, paths, config):
        self.train = json.load(open(paths['data_manifest']))
        self.val   = json.load(open(paths['val_data_manifest']))

        self.config = config

        self.name = "anc"

        self.data_path = config['paths']['data']

        self.train_ids = list(self.train.keys())
        self.val_ids   = list(self.val.keys())

        self.c3d_feats   = h5.File(config.paths["c3d_feats"],'r')
        self.dense_feats = h5.File(config.paths["xception_feats"],'r')
        self.xc_feats    = h5.File(config.paths["xc_feats_val"],'r')

    
        with open("/home/balkhamissi/projects/fmn2/props.json", 'r') as f:
            self.proposal_from_sst = json.loads(f.read())

        
        # self.train_ids  = json.load(open(paths["video_ids"]))["training"]
        # self.val_ids    = json.load(open(paths["video_ids"]))["validation"]

        self.duration = {}
        self.captions = {}
        self.val_captions = {}
        self.gt_intervals = {}
        self.extensions = None


        train_ids = []
        for vid in self.train_ids:
            if vid in self.c3d_feats and vid in self.dense_feats:    
                if abs(self.c3d_feats[vid]['c3d_features'].shape[0] - self.dense_feats[vid].shape[0]) <= 2:
                    train_ids += [vid]
        
        print(f"[INFO] Removed {len(self.train_ids)-len(train_ids)}/{len(self.train_ids)} from training videos")
        self.train_ids = train_ids
        self.val_ids = list(self.proposal_from_sst.keys())


        val_ids = []
        for vid in self.val_ids:
            if vid in self.c3d_feats and vid in self.xc_feats:    
                if abs(self.c3d_feats[vid]['c3d_features'].shape[0] - self.xc_feats[vid].shape[0]) <= 2:
                    val_ids += [vid]

        print(f"[INFO] Removed {len(self.val_ids)-len(val_ids)}/{len(self.val_ids)} from validation videos")
        self.val_ids = val_ids

        print(f"[INFO] # of Training: {len(self.train_ids)} | Validation: {len(self.val_ids)}")

        if self.data_path != "":
            file_names = os.listdir(paths['data'])
            file_names = [f.split('.') for f in file_names]
            self.extensions = { f[0]: f[-1] for f in file_names } ## <vid_id> { . <anything> } . <ext>

        for vid in self.train_ids:
            self.duration[vid] = self.train[vid]['duration']
            self.captions[vid] = self.train[vid]['sentences']
            self.gt_intervals[vid] = self.train[vid]['timestamps']


        for vid in self.val_ids:
            self.duration[vid] = self.val[vid]['duration']
            # self.gt_intervals[vid] = self.val[vid]['timestamps']
            self.gt_intervals[vid] = self.proposal_from_sst[vid]
            self.val_captions[vid] = self.val[vid]['sentences']

        self.vocab_size = None
        self.idx2word = {}
        self.word2idx = {}

        preprocess_captions(self, word_count_threshold=config['caption']['word_count_threshold'])

    def torch_training_raw(self):
        fmn = FMN_Dataset(self, self.train_ids, self.config)
        fmn.pop_element = fmn.raw_and_labels
        
        data_loader_args = {
            'num_workers': self.config['training']['nthreads'],
            'batch_size': self.config['training']['batch_size'],
            'shuffle': self.config['training']['shuffle'],
        }
        return DataLoader(fmn, **data_loader_args, collate_fn=fmn.collate_fn_feats)

    def torch_training_feats(self):
        fmn = FMN_Dataset(self, self.train_ids, self.config)
        fmn.pop_element = fmn.feats_and_labels
        data_loader_args = {
            'num_workers': self.config['training']['nthreads'],
            'batch_size': self.config['training']['batch_size'],
            'shuffle': self.config['training']['shuffle'],
        }
        return DataLoader(fmn, **data_loader_args, collate_fn=fmn.collate_fn_feats)

    def torch_evaluate_feats(self):
        val_ids = np.array(self.val_ids)
        # val_len = len(val_ids)
        # print(f"[INFO] Validating on {int(val_len*0.2)} videos")
        # val_ids = val_ids[np.random.choice(range(val_len), int(val_len*0.2))]
        fmn = FMN_Dataset(self, self.val_ids, self.config)
        fmn.pop_element = fmn.feats_and_labels
        data_loader_args = {
            'num_workers': self.config['training']['nthreads'],
            'batch_size': self.config['training']['batch_size'],
            'shuffle': self.config['training']['shuffle'],
        }
        return DataLoader(fmn, **data_loader_args, collate_fn=fmn.collate_fn_feats)

class ANC_Dataset_CAPTIONS:
    def __init__(self, paths, config):
        self.train = json.load(open(paths['data_manifest']))
        self.val   = json.load(open(paths['val_data_manifest']))

        self.config = config

        self.name = "anc"

        self.data_path = config['paths']['data']

        self.train_ids = list(self.train.keys())
        self.val_ids   = list(self.val.keys())

        with open("/home/balkhamissi/projects/fmn2/props.json", 'r') as f:
            self.proposal_from_sst = json.loads(f.read())
    
        self.duration = {}
        self.captions = {}
        self.val_captions = {}
        self.gt_intervals = {}
        self.extensions = None

        
        self.val_ids = list(self.proposal_from_sst.keys())
        for vid in self.train_ids:
            self.duration[vid] = self.train[vid]['duration']
            self.captions[vid] = self.train[vid]['sentences']
            self.gt_intervals[vid] = self.train[vid]['timestamps']

        for vid in self.val_ids:
            self.duration[vid] = self.val[vid]['duration']
            self.gt_intervals[vid] = self.proposal_from_sst[vid]
            self.val_captions[vid] = self.val[vid]['sentences']

        self.vocab_size = None
        self.idx2word = {}
        self.word2idx = {}

        preprocess_captions(self, word_count_threshold=config['caption']['word_count_threshold'])

    def torch_training_raw(self):
        fmn = FMN_Dataset(self, self.train_ids, self.config)
        fmn.pop_element = fmn.raw_and_labels
        
        data_loader_args = {
            'num_workers': self.config['training']['nthreads'],
            'batch_size': self.config['training']['batch_size'],
            'shuffle': self.config['training']['shuffle'],
        }
        return DataLoader(fmn, **data_loader_args, collate_fn=fmn.collate_fn_feats)

    def torch_training_feats(self):
        fmn = FMN_Dataset(self, self.train_ids, self.config)
        fmn.pop_element = fmn.feats_and_labels
        data_loader_args = {
            'num_workers': self.config['training']['nthreads'],
            'batch_size': self.config['training']['batch_size'],
            'shuffle': self.config['training']['shuffle'],
        }
        return DataLoader(fmn, **data_loader_args, collate_fn=fmn.collate_fn_feats)

    def torch_evaluate_feats(self):
        val_ids = np.array(self.val_ids)
        # val_len = len(val_ids)
        # print(f"[INFO] Validating on {int(val_len*0.2)} videos")
        # val_ids = val_ids[np.random.choice(range(val_len), int(val_len*0.2))]
        fmn = FMN_Dataset(self, val_ids, self.config)
        fmn.pop_element = fmn.feats_and_labels
        data_loader_args = {
            'num_workers': self.config['training']['nthreads'],
            'batch_size': self.config['training']['batch_size'],
            'shuffle': self.config['training']['shuffle'],
        }
        return DataLoader(fmn, **data_loader_args, collate_fn=fmn.collate_fn_feats)
    

excluded_chars = {ord(c): None for c in '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'}

def preprocess_captions(dataset, word_count_threshold=2):
    print('[INFO] Preprocessing word counts and creating vocab based on word count threshold {}'.format(word_count_threshold))        
    word_counts = {}
    
    for _, caps in dataset.captions.items():
        for sent in caps:
            words = sent.lower().translate(excluded_chars).split(" ")
            for w in words:
                word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

    print('[INFO] Filtered words from %d to %d' % (len(word_counts), len(vocab)))

    dataset.idx2word[0] = '<pad>'
    dataset.idx2word[1] = '<bos>'
    dataset.idx2word[2] = '<eos>'
    dataset.idx2word[3] = '<unk>'

    dataset.word2idx['<pad>'] = 0
    dataset.word2idx['<bos>'] = 1
    dataset.word2idx['<eos>'] = 2
    dataset.word2idx['<unk>'] = 3

    dataset.vocab_size = len(vocab) + 4

    for idx, w in enumerate(vocab):
        dataset.word2idx[w] = idx+4
        dataset.idx2word[idx+4] = w
