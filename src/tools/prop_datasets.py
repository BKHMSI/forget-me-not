import json
import os

import h5py
import numpy as np
import progressbar
import torch
from tools import utils

from torch.utils.data import Dataset


class ProposalDataset(object):
    """
    All dataset parsing classes will inherit from this class.
    """

    def __init__(self, config):
        """
        config must contain the following:
            data - the file that contains the Activity Net json data.
            features - the location of where the PCA C3D 500D features are.
        """
        assert os.path.exists(config["train_json"])
        assert os.path.exists(config["val_json"])
        assert os.path.exists(config["features"])

        self.train = json.load(open(config["train_json"]))
        self.val   = json.load(open(config["val_json"]))
        self.testing_ids  = json.load(open(config["test_json"]))

        self.features = h5py.File(config["features"],"r")

        generate_labels = not os.path.exists(config["labels"])
        generate_labels = generate_labels or not os.path.exists(config["vid_ids"])
        
        if  generate_labels :
            exit()
            self.generate_labels(config)

        self.labels  = h5py.File(config["labels"],"r")
        self.vid_ids = json.load(open(config["vid_ids"]))

    def generate_labels(self, args):
        pass
    
    
    def timestamp_to_featstamp(self, timestamp, nfeats, duration):
        start, end = timestamp
        start = min(int(round(start / duration * nfeats)), nfeats - 1)
        end = max(int(round(end / duration * nfeats)), start + 1)
        return start, end

    def compute_proposals_stats(self, prop_captured):
        """
        Function to compute the proportion of proposals captured during labels generation.
        :param prop_captured: array of length nb_videos
        :return:
        """
        nb_videos = len(prop_captured)
        proportion = np.mean(prop_captured[prop_captured != -1])
        nb_no_proposals = (prop_captured == -1).sum()
        print("Number of videos in the dataset: {}".format(nb_videos))
        print("Proportion of videos with no proposals: {}".format(1. * nb_no_proposals / nb_videos))
        print("Proportion of action proposals captured during labels creation: {}".format(proportion))

# For SST (sometimes)
class ActivityNetCaptions(ProposalDataset):
    """
    ActivityNet is responsible for parsing the raw activity net dataset and converting it into a
    format that DataSplit (defined below) can use. This level of abstraction is used so that
    DataSplit can be used with other dataset and we would only need to write a class similar
    to this one.
    """

    def __init__(self, config):
        super(self.__class__, self).__init__(config)
        self.durations = {}
        self.gt_times  = {}
        self.captions  = {}
        self.pos_prop_weight = self.vid_ids['pos_prop_weight']
        self.training_ids = self.vid_ids['training']
        self.validation_ids = self.vid_ids['validation']


        for vid_id in self.training_ids:
            self.durations[vid_id] = self.train[vid_id]["duration"]
            self.gt_times[vid_id]  = self.train[vid_id]["timestamps"]
            self.captions[vid_id]  = self.train[vid_id]["sentences"]
        
        for vid_id in self.validation_ids:
            try:
                self.durations[vid_id] = self.val[vid_id]["duration"]
                self.gt_times[vid_id]  = self.val[vid_id]["timestamps"]
                self.captions[vid_id]  = self.val[vid_id]["sentences"]
            except KeyError as error:
                self.validation_ids.remove(vid_id)

    def generate_labels(self, args):
        """
        Overwriting parent class to generate action proposal labels
        """

        self.training_ids = list(self.train.keys())
        self.validation_ids = list(self.val.keys())

        print("| Generating labels for action proposals")
        label_dataset = h5py.File(args.labels, 'w')
        prop_captured = []
        prop_pos_examples = []

        data = []
        data.extend(self.training_ids)
        data.extend(self.validation_ids)
        train_sz = len(self.training_ids)

        bar = progressbar.ProgressBar(maxval=len(data)).start()

        for progress, video_id in enumerate(data):
            features = self.features[video_id]['c3d_features']
            nfeats   = features.shape[0]
            if progress < train_sz:
                duration   = self.train[video_id]['duration']
                timestamps = self.train[video_id]['timestamps']
            else:
                duration   = self.val[video_id]['duration']
                timestamps = self.val[video_id]['timestamps']

            featstamps = [self.timestamp_to_featstamp(x, nfeats, duration) for x in timestamps]
            nb_prop = len(featstamps)

            for i in range(nb_prop):
                if (featstamps[nb_prop - i - 1][1] - featstamps[nb_prop - i - 1][0]) > args.K / args.iou_threshold:
                    # we discard these proposals since they will not be captured for this value of K
                    del featstamps[nb_prop - i - 1]
            if len(featstamps) == 0:
                if len(timestamps) == 0:
                    # no proposals in this video
                    prop_captured += [-1.]
                else:
                    # no proposals captured in this video since all have a length above threshold
                    prop_captured += [0.]
                if progress < train_sz:
                    self.training_ids.remove(video_id)
                else:
                    self.validation_ids.remove(video_id)
                continue

            labels = np.zeros((nfeats, args.K))
            gt_captured = []
            for t in range(nfeats):
                for k in range(args.K):
                    iou, gt_index = utils.iou([t - k, t + 1], featstamps, return_index=True)
                    if iou >= args.iou_threshold:
                        labels[t, k] = 1
                        gt_captured += [gt_index]
            prop_captured += [1. * len(np.unique(gt_captured)) / len(timestamps)]

            if progress < train_sz:
                prop_pos_examples += [np.sum(labels, axis=0) * 1. / nfeats]

            video_dataset = label_dataset.create_dataset(video_id, (nfeats, args.K), dtype='f')
            video_dataset[...] = labels
            bar.update(progress)

        split_ids = {
            'training': self.training_ids,
            'validation': self.validation_ids,
            'testing': self.testing_ids,
            'pos_prop_weight': np.array(prop_pos_examples).mean(axis=0).tolist() # this will be used to compute the loss
        }

        json.dump(split_ids, open(args.vid_ids, 'w'))
        self.compute_proposals_stats(np.array(prop_captured))
        bar.finish()

class DataProp(Dataset):
    def __init__(self, video_ids, dataset, config):
        """
        video_ids - list of video ids in the split
        features - the h5py file that contain all the C3D features for all the videos
        labels - the h5py file that contain all the proposals labels (0 or 1 per time step)
        args.W - the size of the window (the number of RNN steps to use)
        args.K - The number of proposals per time step
        args.max_W - the maximum number of windows to pass to back
        args.num_samples (optional) - contains how many of the videos in the list to use
        """
        self.video_ids = video_ids
        self.features = dataset.features
        self.labels = dataset.labels
        self.durations = dataset.durations
        self.gt_times = dataset.gt_times
        self.num_samples = config["num_samples"] if "num_samples" in config else None
        self.W = config["W"]
        self.K = config["K"]
        self.max_W = config["max_W"]

        # Precompute masks
        self.masks = np.zeros((self.max_W, self.W, self.K))
        for index in range(self.W):
            self.masks[:, index, :min(self.K, index)] = 1
        self.masks = torch.FloatTensor(self.masks)

    def __getitem__(self, index):
        pass

    def __len__(self):
        if self.num_samples is not None:
            # in case num sample is greater than the dataset itself
            return min(self.num_samples, len(self.video_ids))
        return len(self.video_ids)

class TrainProp(DataProp):
    def __init__(self, video_ids, dataset, args):
        super(self.__class__, self).__init__(video_ids, dataset, args)

    def collate_fn(self, data):
        features = [d[0] for d in data]
        masks    = [d[1] for d in data]
        labels   = [d[2] for d in data]
        return torch.cat(features, 0), torch.cat(masks, 0), torch.cat(labels, 0)

    def __getitem__(self, index):
        # Now let's get the video_id
        video_id = self.video_ids[index]
        features = self.features[video_id]['c3d_features']
        labels = self.labels[video_id]
        nfeats = features.shape[0]
        nWindows = max(1, nfeats - self.W + 1)

        # Let's sample the maximum number of windows we can pass back.
        sample = range(nWindows)
        if self.max_W < nWindows:
            # bkhmsi - take random start points of windows
            sample = np.random.choice(nWindows, self.max_W)
            nWindows = self.max_W

        # Create the outputs
        masks = self.masks[:nWindows, :, :]
        feature_windows = np.zeros((nWindows, self.W, features.shape[1]))
        label_windows = np.zeros((nWindows, self.W, self.K))
        for j, w_start in enumerate(sample):
            # bkhmsi - moving with stride 1
            w_end = min(w_start + self.W, nfeats)
            feature_windows[j, 0:w_end - w_start, :] = features[w_start:w_end, :]
            label_windows[j, 0:w_end - w_start, :] = labels[w_start:w_end, :]
        return torch.FloatTensor(feature_windows), masks, torch.Tensor(label_windows)

class ValidProp(DataProp):
    def __init__(self, video_ids, dataset, args):
        super(self.__class__, self).__init__(video_ids, dataset, args)

    def collate_fn(self, data):
        features = [d[0] for d in data]
        masks    = [d[1] for d in data]
        labels   = [d[2] for d in data]
        return torch.cat(features, 0), torch.cat(masks, 0), torch.cat(labels, 0)

    def __getitem__(self, index):
        # Now let's get the video_id
        video_id = self.video_ids[index]
        features = self.features[video_id]['c3d_features']
        duration = self.durations[video_id]
        gt_times = self.gt_times[video_id]
        labels = self.labels[video_id]
        
        n_vids = len(gt_times)

        feature_windows = np.zeros((n_vids, self.W, features.shape[-1]))
        label_windows = np.zeros((n_vids, self.W, self.K))


        for idx in range(n_vids):
            w_start, w_end = self.timestamp_to_featstamp(gt_times[idx], 
               feature_windows.shape[1], duration)
        
            if w_start > w_end:
                w_start, w_end = w_end, w_start

            w_end = min(w_end, features.shape[0])
            prop_len = min(self.W, w_end - w_start)
            
            if features[w_start:w_end, :].shape[0] <= 0:
                continue

            feature_windows[idx, 0:(w_end - w_start), :]  = features[w_start:w_end, :]
            label_windows[idx, 0:(w_end - w_start), :] = labels[w_start:w_end, :]
        
        return torch.FloatTensor(feature_windows), self.masks[:n_vids, :, :], torch.Tensor(label_windows)

    def timestamp_to_featstamp(self, timestamp, nfeats, duration):
        start, end = timestamp
        start = min(int(round(start / duration * nfeats)), nfeats - 1)
        end = max(int(round(end / duration * nfeats)), start + 1)
        return start, end

class EvaluateProp(DataProp):
    def __init__(self, video_ids, dataset, args):
        super(self.__class__, self).__init__(video_ids, dataset, args)

    def collate_fn(self, data):
        features  = data[0][0]
        gt_times  = data[0][1]
        durations = data[0][2]
        video_ids = data[0][3]
        return features.view(1, features.size(0), features.size(1)), gt_times, durations, video_ids

    def __getitem__(self, index):
        # Let's get the video_id and the features and labels
        video_id = self.video_ids[index]
        features = self.features[video_id]['c3d_features']
        duration = self.durations[video_id]
        gt_times = self.gt_times[video_id]

        return torch.FloatTensor(features), gt_times, duration, video_id

