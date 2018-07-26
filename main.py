import os
import argparse

import toml

import torch as T

from bunch import Bunch

import src.pipelines as models
from src.pipelines import rc_softcap as rcsc
from src.pipelines import dc_softcap as dcsc
from src.pipelines import xc_s2vt    as xcst
from src.tools import datasets, utils as us


def main(args):
    config = toml.load(args.config)
    config = Bunch(config)
    paths  = config['paths']

    if not os.path.isdir(paths["weights_save"]):
        os.makedirs(paths["weights_save"])

    if "softcap" in config.run_title:
        model = dcsc.DC_SoftCap
    else:
        model  = xcst.XC_S2VT

    if config.run_title[-5:] == "lsmdc":
        dataset = datasets.LSMDC_Dataset
    else:
        dataset = datasets.ANC_Dataset

    dataset = dataset(paths, config)
    optim  = lambda w_params: T.optim.Adam(w_params, lr=config.training["lr"], weight_decay=config.training["weight_decay"])
    # optim = lambda w_params: T.optim.SGD(w_params, lr=config.training["lr"], momentum=config.training["momentum"], nesterov=True, weight_decay=config.training["weight_decay"])
    params = config['training']

    T.cuda.manual_seed(params["seed"])
    trainer = us.Trainer(model, dataset, config, optim)
    if args.mode == 0:
        trainer.train(params)  
    elif args.mode == 1:
        trainer.evaluate()
    elif args.mode == 2:
        trainer.eval_loss(params)
    
def get_args():
    parser = argparse.ArgumentParser(description="Our awesome baTee5...")

    parser.add_argument('--cuda'  , '-g', type=int, default=0)
    parser.add_argument('--mode'  , '-m', type=int, default=0)
    parser.add_argument('--config', '-c', type=str,
                        metavar='CONFIG.TOML', help='Path to configuration toml file')

    return parser.parse_args()

if __name__ == '__main__':
    print('[INFO] Running as main script')
    args = get_args()
    with T.cuda.device(args.cuda):
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        main(args)
