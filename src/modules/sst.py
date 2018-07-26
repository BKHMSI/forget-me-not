import h5py as h5
import numpy as np

import torch
import torch.nn as nn


class SST(nn.Module):
    """
    Container module with 1D convolutions to generate proposals
    """

    def __init__(self, config):
        super(SST, self).__init__()
        self.rnn = getattr(nn, config["rnn_type"])(
            config["video_dim"],
            config["hidden_dim"],
            config["rnn_num_layers"],
            batch_first=True,
            dropout=config["rnn_dropout"])

        #self.scores_dropout = torch.nn.Dropout(config["dropout_out"])
        # adding dropout
        self.scores = torch.nn.Linear(config["hidden_dim"], 
            config["K"])

        # Saving arguments
        self.video_dim = config["video_dim"]
        self.W = config["W"]
        self.rnn_type = config["rnn_type"]
        self.rnn_num_layers = config["rnn_num_layers"]
        self.rnn_dropout = config["rnn_dropout"]
        self.K = config["K"]

    def network_mode(self, training):
        if training:
            # self.rnn.dropout = self.rnn_dropout
            self.train()
        else:
            self.eval()

    def forward(self, features):
        N, T, _ = features.size()

        rnn_output, _ = self.rnn(features)
        rnn_output = rnn_output.contiguous()
        rnn_output = rnn_output.view(rnn_output.size(0) * rnn_output.size(1), rnn_output.size(2))
        
        #scores = self.scores(self.scores_dropout(rnn_output))
        scores = self.scores(rnn_output)
        outputs = torch.sigmoid(scores)
        
        return outputs.view(N, T, self.K)

    def compute_loss(self, outputs, masks, labels):
        """
        Our implementation of weighted BCE loss.
        """
        labels = labels.view(-1)
        masks = masks.view(-1)
        outputs = outputs.view(-1)

        # Generate the weights
        ones = torch.sum(labels)
        total = labels.nelement()
        weights = torch.FloatTensor(outputs.size()).type_as(outputs.data)
        weights[labels.long() == 1] = 1.0 - ones / total
        weights[labels.long() == 0] = ones / total
        weights = weights.view(weights.size(0), 1).expand(weights.size(0), 2)

        # Generate the log outputs
        outputs = outputs.clamp(min=1e-8)
        log_outputs = torch.log(outputs)
        neg_outputs = 1.0 - outputs
        neg_outputs = neg_outputs.clamp(min=1e-8)
        neg_log_outputs = torch.log(neg_outputs)
        all_outputs = torch.cat((log_outputs.view(-1, 1), neg_log_outputs.view(-1, 1)), 1)

        all_values = all_outputs.mul(torch.autograd.Variable(weights))
        all_labels = torch.autograd.Variable(torch.cat((labels.view(-1, 1), (1.0 - labels).view(-1, 1)), 1))
        all_masks = torch.autograd.Variable(torch.cat((masks.view(-1, 1), masks.view(-1, 1)), 1))
        loss = -torch.sum(all_values.mul(all_labels).mul(all_masks)) / outputs.size(0)
        
        return loss

    def compute_loss_with_BCE(self, outputs, masks, labels, w1):
        """
        Uses weighted BCE to calculate loss
        """
        w1 = torch.FloatTensor(w1).type_as(outputs.data)
        w0 = 1. - w1
        labels = labels.mul(masks)
        weights = labels.mul(w0.expand(labels.size())) + (1. - labels).mul(w1.expand(labels.size()))
        weights = weights.view(-1)
        labels = torch.autograd.Variable(labels.view(-1))
        masks = torch.autograd.Variable(masks.view(-1))
        outputs = outputs.view(-1).mul(masks)
        criterion = torch.nn.BCELoss(weight=weights)
        loss = criterion(outputs, labels)
        return loss


    @staticmethod
    def get_segments(y, stride=8):
        """Convert predicted output tensor (y_pred) from SST model into the
        corresponding temporal proposals. Can perform standard confidence
        thresholding/post-processing (e.g. non-maximum suppression) to select
        the top proposals afterwards.
        Parameters
        ----------
        y : ndarray
            Predicted output from SST model of size (L, K), where L is the length of
            the input video in terms of discrete time steps.
        stride : int, optional
            The temporal resolution of the visual encoder in terms of frames. See
            Section 3 of the main paper for additional details.
        Returns
        -------
        props : ndarray
            Two-dimensional array of shape (num_props, 2), containing the start and
            end boundaries of the temporal proposals in units of frames.
        scores : ndarray
            One-dimensional array of shape (num_props,), containing the
            corresponding scores for each detection above.
        """
        y = y.data.cpu().numpy()
        temp_props, temp_scores = [], []
        L, K = y.shape
        for i in range(L):
            for j in range(min(i+1, K)):
                #removed strides to return feat_stamps rather than frame_stamps
                temp_props.append([(i-j-1), i])
                temp_scores.append(y[i, j])
        props_arr, score_arr = np.array(temp_props), np.array(temp_scores)
        # filter out proposals that extend beyond the start of the video.
        idx_valid = props_arr[:, 0] >= 0
        props, scores = props_arr[idx_valid, :], score_arr[idx_valid]
        return props, scores

    @staticmethod
    def nms_detections(props, scores, overlap=0.5):
        """Non-maximum suppression: Greedily select high-scoring detections and
        skip detections that are significantly covered by a previously selected
        detection. This version is translated from Matlab code by Tomasz
        Malisiewicz, who sped up Pedro Felzenszwalb's code.
        Parameters
        ----------
        props : ndarray
            Two-dimensional array of shape (num_props, 2), containing the start and
            end boundaries of the temporal proposals.
        scores : ndarray
            One-dimensional array of shape (num_props,), containing the corresponding
            scores for each detection above.
        Returns
        -------
        nms_props, nms_scores : ndarrays
            Arrays with the same number of dimensions as the original input, but
            with only the proposals selected after non-maximum suppression.
        """
        t1 = props[:, 0]
        t2 = props[:, 1]
        ind = np.argsort(scores)
        area = (t2 - t1 + 1).astype(float)
        pick = []
        while len(ind) > 0:
            i = ind[-1]
            pick.append(i)
            ind = ind[:-1]
            tt1 = np.maximum(t1[i], t1[ind])
            tt2 = np.minimum(t2[i], t2[ind])
            wh = np.maximum(0., tt2 - tt1 + 1.0)
            o = wh / (area[i] + area[ind] - wh)
            ind = ind[np.nonzero(o <= overlap)[0]]
        nms_props, nms_scores = props[pick, :], scores[pick]
        return nms_props, nms_scores
