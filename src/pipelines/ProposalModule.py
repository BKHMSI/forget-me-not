import os
import numpy as np
import pandas as pd
import toml
from bunch import Bunch
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tools.prop_datasets import TrainProp, ActivityNetCaptions, EvaluateProp, ValidProp
from modules.sst import *

import json

class ProposalModule(object):
    def __init__(self, config):
        self.config = config
        self.model_save_dir = os.path.join(self.config["save_dir"], 
            "{:%B_%d_%Y_time_%H_%M}".format(datetime.now()))
        
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

    def adjust_lr(self, epoch):
        lr = self.config["lr"] * (0.2 ** (epoch // 5))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self):
        self.optimizer = optim.SGD(
             self.model.parameters(),
             lr=0.01, momentum=0.9,
             weight_decay=0,
             nesterov=False)
        #self.optimizer = optim.Adam(
        # self.model.parameters(), 
        # lr=self.config["lr"], 
        # weight_decay=self.config["weight_decay"])
        # self.optimizer = optim.Adamax(self.model.parameters(), 
        # weight_decay = self.config["weight_decay"])
        #scheduler = ReduceLROnPlateau(self.optimizer, 'min')

        for epoch in range(self.config["epochs"]):
            #self.adjust_lr(epoch)
            #print(f"Starting epoch {epoch}")
            train_loss = self.train_epoch(epoch)
            #print(f"Starting validation of epoch {epoch}")
            valid_loss = self.validate(epoch)

            #print('Finished epoch {}\n'.format(epoch))
            

            epoch_results = f"Epoch:\t{epoch}\n"
            epoch_results += f"train_loss:\t{train_loss}\nvalid_loss:{valid_loss}"
            print(f"epoch:\t{epoch}\ttrain_loss:\t{train_loss:3.5}\tvalid_loss:\t{valid_loss:3.5}")

            log_path = os.path.join(self.model_save_dir,"logs.txt")
            with open(log_path,'a') as file:
                file.write(epoch_results)            
            
            self.save_model(epoch)

    def save_model(self, epoch):
        path = os.path.join(self.model_save_dir, "model_{}".format(epoch))
        torch.save(self.model.state_dict(), path)

    def train_epoch(self, epoch):        
        self.model.network_mode(training=True)

        overall_loss = 0.0
        for batch_idx, (feats, masks, labels) in enumerate(self.train_loader):
            feats = Variable(feats.cuda())
            masks = masks.cuda()
            labels = labels.cuda()

            self.optimizer.zero_grad()

            props = self.model(feats)
            batch_loss = self.model.compute_loss(
                props, masks, labels)#, self.pos_prop_weight
            batch_loss.backward()
            self.optimizer.step()

            overall_loss += batch_loss.data[0]
            print(f" " * 80, end='\r') 
            print(f"epoch:\t{epoch}\tloss:\t{overall_loss/(1+batch_idx):3.4}\tBatch:\t{batch_idx}", end='\r')    
        
        print('') # just to print new line

        overall_loss /= batch_idx
        return overall_loss

    def validate(self, epoch):
        self.model.network_mode(training=False)

        overall_loss = 0.0
        for batch_idx, (feats, masks, labels) in enumerate(self.val_evaluator):
            feats = Variable(feats.cuda())
            masks = masks.cuda()
            labels = labels.cuda()

            props = self.model(feats)
            batch_loss = self.model.compute_loss(
                props, masks, labels)#, self.pos_prop_weight

            overall_loss += batch_loss.data[0]
        
        overall_loss /= batch_idx
        return overall_loss
    
    def evaluate(self, maximum=None):
        output = {}
        
        self.model.eval()
        total = len(self.eval_evaluator)
        if maximum:
            total = min(total, maximum)
        recall = np.zeros(total)


        print(f"maximum {maximum}")
        exit()

        video_duration = []
        feat_count = []

        all_gt_feats = []
        gt_video_names = []
        proposals  = [None] * total
        video_name = [None] * total
        # try:
        props_count_noob = []
        for batch_idx, (features, gt_times, duration, vid_id) in enumerate(self.eval_evaluator):
            if maximum is not None and batch_idx >= maximum:
                break
            feat_len = features.shape[1]
            #print('num feats',feat_len)
            
            features = features.cuda()
            features = Variable(features)
            # proposals: (1, T, K)
            # y_pred, hidden = self.model(features)
            y_pred = self.model(features)
            props_raw, scores_raw = SST.get_segments(y_pred[0, :, :])
            props, scores = SST.nms_detections(props_raw, scores_raw, self.config["nms_thresh"])
            n_prop_after_pruning = min(1000, scores.size)

            
            props = props[:n_prop_after_pruning]
            scores = scores[:n_prop_after_pruning]

            
            


            props_count_noob.append(n_prop_after_pruning)

            # #print('gt_times len ',len(gt_times))
            for gt_time in gt_times:
                #print(gt_time)
               # print([timestamp_to_featstamp(gt_time, feat_len, duration)])
                all_gt_feats.append([self.timestamp_to_featstamp(gt_time, feat_len, duration)]) 
                gt_video_names.append(vid_id)


            video_duration.extend([duration] * n_prop_after_pruning)
            feat_count.extend([feat_len] * n_prop_after_pruning)

            
            proposals[batch_idx] = np.hstack([
                props, scores.reshape((-1, 1)),
                np.zeros((n_prop_after_pruning, 1))])
            video_name[batch_idx] = np.repeat([vid_id], n_prop_after_pruning).reshape(
                n_prop_after_pruning, 1)

        #props_count_noob = np.array(props_count_noob).mean()
        #print("mean props ", props_count_noob)

        gt_video_names = np.array(gt_video_names).reshape(-1, 1)
        video_duration = np.array(video_duration).reshape(-1, 1)
        feat_count = np.array(feat_count).reshape(-1, 1)

        all_gt_feats = np.vstack(all_gt_feats)
        proposals_arr = np.vstack(proposals)
        proposals_vid = np.vstack(video_name)
        
        #output_file_prop = os.path.join("/home/balkhamissi/projects/fmn/dataframe", "eval_prop_out_rand.out")
        #output_file_gt = os.path.join("/home/balkhamissi/projects/fmn/dataframe", "ground_truth_rand.out")
        

        #output_file = os.path.join(args.output_dir, args.output_name)
        # df = pd.concat([
        #     pd.DataFrame(proposals_arr[:, 0:2] * video_duration / feat_count,
        #      columns=['time-init','time-end']),
        #     pd.DataFrame(proposals_arr, columns=['f-init', 'f-end', 'score',
        #                                         'video-frames']),
        #     pd.DataFrame(proposals_vid, columns=['video-name'])],
        #     axis=1)
        # df.to_csv(output_file_prop, index=None, sep=' ')

        # df = pd.concat([
        #     pd.DataFrame(all_gt_feats, columns=['f-init', 'f-end']),
        #     pd.DataFrame(gt_video_names, columns=['video-name'])],
        #     axis=1)
        # df.to_csv(output_file_gt, index=None, sep=' ')

        with open(self.config["demo_props"],'w') as f:
            f.write(json.dumps(output))

        
    def timestamp_to_featstamp(self, timestamp, nfeats, duration):
        start, end = timestamp
        start = min(int(round(start / duration * nfeats)), nfeats - 1)
        end = max(int(round(end / duration * nfeats)), start + 1)
        return start, end

    def load_data(self, dataset):
        self.pos_prop_weight = dataset.pos_prop_weight

        val_size = int(len(dataset.validation_ids) * 0.2)


        train_dataset = TrainProp(dataset.training_ids,
                dataset, self.config)
        val_dataset = ValidProp(dataset.validation_ids[:val_size], 
                dataset, self.config)

        eval_dataset = EvaluateProp(dataset.validation_ids,
            dataset, self.config)

        self.train_loader  = DataLoader(train_dataset, 
                shuffle=self.config["shuffle"], 
                batch_size=self.config["batch_size"], 
                num_workers=self.config["nthreads"], 
                collate_fn=train_dataset.collate_fn)

        self.val_evaluator = DataLoader(val_dataset,   
                shuffle=self.config["shuffle"], 
                batch_size=1, 
                num_workers=self.config["nthreads"], 
                collate_fn=val_dataset.collate_fn)

        self.eval_evaluator = DataLoader(eval_dataset,   
                shuffle=self.config["shuffle"], 
                batch_size=1, 
                num_workers=self.config["nthreads"], 
                collate_fn=eval_dataset.collate_fn)

    def load_model(self):
        self.model = SST(self.config)

        if config["load_prop"] == True:
            print("Loading weights")
            self.model.load_state_dict(torch.load(config["load_prop_path"]))

        self.model.cuda()


if __name__ == '__main__':
    TOML_FILE = "/home/balkhamissi/projects/fmn2/config/SST.toml"
    with open(TOML_FILE) as f:
        config = Bunch(toml.load(f)).proposal

        dataset = ActivityNetCaptions(config)
        
        prop = ProposalModule(config)
        prop.load_model()
        prop.load_data(dataset)
        #prop.train()
        prop.evaluate()


    
