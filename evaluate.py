import os
import argparse
import toml

import torch as T

from bunch import Bunch

import src.pipelines as models
from src.pipelines import demo 
from src.tools import datasets, utils as us


class Evaluate:
    def __init__(self):
        args   = get_args()
        config = toml.load(args.config)
        config = Bunch(config)
        paths  = config['paths']
        
        T.cuda.manual_seed(config.seed)

        dataset = datasets.ANC_Dataset_CAPTIONS(paths, config)

        self.model = demo.Demo_3(dataset, config)

    def main(self, videos, query):
        return self.model.run(videos, query)
    
def get_args():
    parser = argparse.ArgumentParser(description="Our awesome inference baTee5...")

    parser.add_argument('--config', '-c', type=str, default="config/demo-1.toml",
                        metavar='CONFIG.TOML', help='Path to configuration toml file')
    parser.add_argument('--video', '-v', type=str, default="v_D18b2IZpxk0", help='Video ID')
    parser.add_argument('--query', '-q', type=str, default="woman riding camel", help='Query')
    parser.add_argument('--cuda' , '-g', type=int, default=0, help='Cuda')

    return parser.parse_args()

if __name__ == '__main__':
    print('[INFO] Running as main script')
    args = get_args()
    with T.cuda.device(args.cuda):
        main(args.video, args.query)