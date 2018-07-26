import os
import json
from time import time

import numpy as np

import torch as T
import torch.nn as nn

from bisect import bisect_left, bisect_right

class Trainer:
    def __init__(self, model, dataset, config, optim, callbacks=None, model_args={}):
        if callbacks is not None:
            self.cb = callbacks
        else:
            self.cb = {
                'epoch_log': lambda x: print('\r[epoch]', x),
                'batch_log': lambda x: print('\r[batch]', x, end='')
            }

        self.dataset = dataset
        self.model   = model(config, dataset.vocab_size, **model_args)
        self.optim   = optim
        self.config  = config
        
        self.model.init(optim)

    def train(self, params):
        """
        params : Training params
        """
        ## Training setup
        self.params = params

        log_on_epoch = self.cb['epoch_log']
        log_on_batch = self.cb['batch_log']

        start_epoch = params['start_epoch']
        n_epochs    = params['n_epochs']
        batch_size  = params['batch_size']
        run_title   = self.config['run_title']

        training_data = self.model.get_training(self.dataset)
        eval_data = self.model.get_evaluate(self.dataset)
        sched = T.optim.lr_scheduler.ExponentialLR(self.model.optim, gamma=0.98)


        for epoch in range(n_epochs):
            new_epoch = start_epoch + epoch + 1
            lr = self.model.global_lr()
            print(f'epoch {new_epoch}, lr: {lr:2.5g}')

            batch_losses = []
            start_time = time()
            self.model.model.train()
            for ib, xy in enumerate(training_data):
                if xy is None: continue
                
                self.model.cycle(xy)

                loss = self.model.last_loss()
                batch_losses.append(loss)

                progress = ib / len(training_data)
                elapsed = time() - start_time
                mean_loss = np.mean(batch_losses)
                log_line = f'{new_epoch:>4} | {progress:>3.2%} | {elapsed:>4.1f} ms/epoch | train_loss {mean_loss:4.2f}'
                log_on_batch(log_line)

            sched.step()
            
            log_on_epoch(log_line)
            self.model.model.eval()
            self.eval_on_epoch(eval_data, new_epoch)

            if params['checkpoint_interval'] > 0 and epoch % params['checkpoint_interval'] == 0:
                weights_name = run_title + str(new_epoch) + '.pth'
                weights_path = os.path.join(self.config['paths']['weights_save'], weights_name)
                T.save(self.model.model.state_dict(), weights_path)


    def evaluate(self):
        ## Evaluate setup
        self.model.load_model()
        eval_data = self.model.get_evaluate(self.dataset)
        self.model.model.eval()
        for ib, xy in enumerate(eval_data):
            if xy is None: continue
            self.model.evaluate(xy, self.dataset.idx2word)

        with open(f'fmn2/results/{self.config.paths["caption_save"]}.json', 'w') as outfile:
            json.dump({"version": "VERSION 1.0", "results": self.model.results}, outfile)

    def eval_on_epoch(self, eval_data, epoch):
        """
        params : Evaluate params
        """

        log_on_epoch = self.cb['epoch_log']
        log_on_batch = self.cb['batch_log']

        batch_losses = []
        start_time = time()

        for ib, xy in enumerate(eval_data):
            if xy is None: continue
                
            self.model.evaluate(xy)

            loss = self.model.last_loss()
            batch_losses += [loss]

            progress = 100 * ib / len(eval_data)
            elapsed = time() - start_time
            mean_loss = np.mean(batch_losses)
            log_line = f'{epoch:>4} | {progress:>3.1f}% | {elapsed:>4.1f} ms/epoch | val_loss {mean_loss:4.2f}'
            log_on_batch(log_line)

        log_on_epoch(log_line)


    def eval_loss(self, params):
        """
        params : Evaluate params
        """
        ## Training setup
        self.params = params

        log_on_epoch = self.cb['epoch_log']
        log_on_batch = self.cb['batch_log']

        start_epoch = params['start_epoch']
        n_epochs    = params['n_epochs']
        batch_size  = params['batch_size']

        eval_data = self.model.get_evaluate(self.dataset)

        lr = self.model.global_lr()

        n_epochs = 29

        for epoch in range(n_epochs):
            new_epoch = start_epoch + epoch + 1
            print(f'[INFO] model of epoch {new_epoch}')

            batch_losses = []
            start_time = time()
            self.model.load_model(model_path=f"./weights/xc-softcap-2/16-04-ANC/xc-softcap-anc{new_epoch}.pth")

            for ib, xy in enumerate(eval_data):
                if xy is None: continue
                
                self.model.evaluate(xy)

                loss = self.model.last_loss()
                batch_losses += [loss]

                progress = 100 * ib / len(eval_data)
                elapsed = time() - start_time
                mean_loss = np.mean(batch_losses)
                log_line = f'{new_epoch:>4} | {progress:>3.1f}% | {elapsed:>4.1f} ms/epoch | val_loss {mean_loss:4.2f}'
                log_on_batch(log_line)

            log_on_epoch(log_line)




def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError

def find_lt(a, x):
    'Find rightmost value less than x'
    i = bisect_left(a, x)
    if i:
        return a[i-1]
    raise ValueError

def find_le(a, x):
    'Find rightmost value less than or equal to x'
    i = bisect_right(a, x)
    if i:
        return a[i-1]
    raise ValueError

def find_gt(a, x):
    'Find leftmost value greater than x'
    i = bisect_right(a, x)
    if i != len(a):
        return a[i]
    raise ValueError

def find_ge(a, x):
    'Find leftmost item greater than or equal to x'
    i = bisect_left(a, x)
    if i != len(a):
        return a[i]
    raise ValueError

def timestamp_to_featstamp(timestamp, nfeats, duration):
    """
    Function to measure 1D overlap
    Convert the timestamps to feature indices
    """
    start, end = timestamp
    start = min(int(round(start / duration * nfeats)), nfeats - 1)
    end = max(int(round(end / duration * nfeats)), start + 1)
    return start, end
