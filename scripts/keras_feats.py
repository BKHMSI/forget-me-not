import os
import shutil
import sys
from time import time

import cv2 as cv
import h5py as h5
import imageio as iio
import keras as K
from keras.layers import GlobalAveragePooling2D
import numpy as np
from bunch import Bunch
from keras.applications import densenet
from PIL import Image

from fmn2.foreign.casson import c3d, sports1M_utils

out_path = None
in_path = None
list_path = None

class Pipeline:
    im_size = (256, 256)
    model : K.Model

    def predict(self, r):
        return self.model.predict(r)

    def preprocess(self, r):
        return preprocess(self, r)

def npcrop(X, h, w):
    '''
    Expects X : (b, h, w, c)
    '''
    H, W = X.shape[1:3]
    j = (H-h)//2
    i = (W-w)//2
    X = X[:, j:j+h, i:i+w, :]
    return X

def preprocess(pipeline, X):
    csize, mean, std = pipeline.prep
    X = npcrop(X, *csize)
    X = X - mean
    X /= std
    return X

def extract_b(pipeline : Pipeline, vid, vlen_limit=-1, stride=-1):
    '''
    pipeline: Class with Keras Model (model), predict function, preprocess function, scanner generator
    sampling_params: List of 1 or 2, (video_end [, stride]) to sample frames with
                    The resulting sequence (batch) is passed to the pipeline preprocessor
                    and predictor according to its scanner generator
    '''
    vlen = vid.get_length()
    im_size = np.array(pipeline.im_size)

    if vlen_limit == -1:
        vlen_limit = vlen
    if stride == -1:
        stride = 1

    raw = []

    for i in range(0, vlen_limit, stride):
        try:
            r = np.array(vid.get_data(i))
        except:
            vlen = i
            break
        size  = np.array(r.shape[:2])
        fxy   = (im_size / size).max()
        dsize = np.ceil(fxy * size).astype(np.int)
        r = cv.resize(r, tuple(dsize), interpolation=cv.INTER_AREA)
        # r = np.expand_dims(r, axis=0)
        raw.append(r)

    raw = np.array(raw)
    feats = []

    pscanner = pipeline.scanner(vlen)
    for i in pscanner:
        r = raw[i]
        ## r : (b?, W, h, w, c)
        r = pipeline.preprocess(r)
        featm = pipeline.predict(r)
        feats.append(featm)
    
    feats = np.array(feats).squeeze()
    return feats.shape[0], feats

def extract_f(pipeline : Pipeline, vid, stride):
    vlen = vid.get_length()
    im_size = np.array(pipeline.im_size)

    raw = []

    for i in range(0, vlen, stride):
        try:
            r = np.array(vid.get_data(i))
        except:
            vlen = i
            break

        size  = np.array(r.shape[:2])
        fxy   = (im_size / size).max()
        dsize = np.ceil(fxy * size).astype(np.int)
        r = cv.resize(r, tuple(dsize), interpolation=cv.INTER_AREA)
        r = np.expand_dims(r, axis=0)
        raw.append(r)
    
    ## : (b, h, w, c)
    raw = np.concatenate(raw, axis=0)
    raw = pipeline.preprocess(raw)

    return vlen, pipeline.predict(raw)

class Xception(Pipeline):
    im_size = (324, 324)
    def __init__(self):
        self.model = K.applications.Xception(include_top=False, pooling='avg')

        x = self.model.get_layer('block14_sepconv1_act').output
        x = GlobalAveragePooling2D()(x)

        self.model = K.Model(inputs=self.model.input, outputs=x)

        self.prep  = ((299, 299), 255/2, 255/2)

    @staticmethod
    def scanner(vlen):
        return [np.s_[...]]

    def preprocess(self, X):
        ## https://github.com/fchollet/deep-learning-models/blob/master/xception.py#L265
        X = X / 255.
        X -= 0.5
        X *= 2
        return X

class DenseNet(Pipeline):
    im_size = (256, 256)
    scanner = lambda _: np.s_[...]
    nets = {
        121: densenet.DenseNet121,
        169: densenet.DenseNet169,
        201: densenet.DenseNet201,
    }
    
    def __init__(self, i=169):
        model = DenseNet.nets[i]
        self.model : K.Model = model(include_top=False, weights='imagenet', pooling='avg')
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        self.prep  = ((224, 224), mean, std)

class C3D(Pipeline):
    def __init__(self):
        model = c3d.C3D()
        self.model = K.Model(inputs=model.input, outputs=model.get_layer('fc6').output)
        self.prep = None

    @staticmethod
    def scanner(vlen):
        return (np.s_[i:i+16] for i in range(0, vlen, 8) if i+16 <= vlen)

    def preprocess(self, X):
        return sports1M_utils.preprocess_input(X)

def main(pipeline : Pipeline, list_path=None, sampling_params=[-1]):
    out_dir = out_path[:out_path.rfind('/')]

    h = h5.File(out_path, 'a')

    fs = sorted(open(list_path, 'r').read().split('\n'))

    # vs = (l.split('/')[-1][:l.rfind('.')] for l in fs)
    vs = [l[:l.rfind('.')] for l in fs]
    # fs = [(v, f) for (v, f) in zip(vs, fs) if v not in h]
    fs = [(v, f) for (v, f) in zip(vs, fs)]

    fs = fs[18:]

    N = len(fs)
    logger = Log(N)
    zb = 0
    errors = open(f'{out_dir}/zbframes.log', 'a')

    errors.write(str(time()))

    print(f"*** Extracting features from {N} videos with model {type(pipeline)} ...")
    n_correct = 0
    try:
        for i, (v, f) in enumerate(fs):
            logger.logj(v, i)

            try:
                video = iio.get_reader(os.path.join(in_path, f), '.mp4')
                _, feats = extract_b(pipeline, video, stride=8)#*sampling_params)
            except:
                # print(f, flush=True)
                errors.write(f"{f}\n")
                errors.flush()
                h.pop(v, None)
                h.flush()
                zb += 1

            logger.logj(v, i)
            print('.', end='')

            try:
                h[v] = feats
            except RuntimeError:
                del h[v]
                h[v] = feats

            if i % 20 == 0 or (N-i) <= 20:
                h.flush()
    finally:
        errors.close()
        h.close()

    print(f'\n*** Done | {zb} clips with 0B frames | {n_correct}/{N} clips skipped')

class Log:
    def __init__(self, n):
        self.n = n

    @staticmethod
    def clear():
        cols = shutil.get_terminal_size()[0]
        print('\r' + (' ' * cols), end='')
    
    @staticmethod
    def println(*w, **kw):
        print('\r', *w, **kw, end='', flush=True)

    def logj(self, i, ii):
        self.clear()
        n = self.n
        self.major = f"{i:19} | {ii}/{n} {ii/n*100:3.1f}% ."
        self.println(self.major)

    def logn(self, i):
        self.minor = f"{self.major} | {i*100:3.1f} ."
        self.println(self.minor)
    
## python keras_feats.py cuda_nums list_path base_path, out_path
if __name__ == '__main__':
    print("*** Running as Main")
    cuda = sys.argv[1]
    if cuda != '-':
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda
    list_path = sys.argv[2]
    in_path   = sys.argv[3]
    out_path  = sys.argv[4]


    # dn = DenseNet()
    # c3d = C3D()
    # sampling_params = [-1]
    xc = Xception()

    main(xc, list_path)
