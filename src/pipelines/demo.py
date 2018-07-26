import json
import numpy as np
# from tinydb import TinyDB, Query

import torch as T
from torch.autograd import Variable

import src.modules.sst  as sst
import src.modules.s2vt as s2vt
import src.modules.softcap as softcap
import src.modules.actionet as actionet
import src.modules.skipthought as skipthought

from nltk.corpus import stopwords

class Demo_1:

    def __init__(self, dataset, config):

        self.index = {} # will contain { interval: caption }
        self.dataset = dataset

        dataset.vocab_size = 12528

        self.c3d   = actionet.ANC_C3D(config)
        self.sst   = sst.SST(config)
        self.s2vt  = s2vt.S2VT(dataset.vocab_size, config)
        # self.skip  = skipthought.SkipThought(dataset, config)

        self.sst.load_state_dict(T.load(config.paths["sst_weights"]))
        self.s2vt.load_state_dict(T.load(config.paths["s2vt_weights"]))

        self.sst.eval()
        self.s2vt.eval()

        self.sst.cuda()
        self.s2vt.cuda()

    def run(self, video, query):
        intervals = self.dataset.gt_intervals[video]
  
        for interval in intervals[:1]:
            feats = self.features(video, interval)[0]
            caption = self.captions(feats)
            print("[DEMO] Caption without Beam: {}".format(caption))
            for i in range(1,5):
                caption = self.captions(feats, i)
                # similarity = self.match_query(caption, query)
                # print("[DEMO] Similarity: {}".format(similarity))
                print("[DEMO] Caption with Beam {}: {}".format(i, caption))

    def match_query(self, caption, query):
        similarity = self.skip(caption, query)
        return similarity

    def captions(self, feats, beam_width = None):
        if beam_width is not None:
            gen_words = self.s2vt.evaluate_beamsearch(Variable(T.FloatTensor(feats)), beam_width)
        else:
            gen_words = self.s2vt.evaluate(Variable(T.FloatTensor(feats)))
        caption   = ' '.join(self.dataset.idx2word[gen] for gen in gen_words)
        return caption

    def proposals(self, feats, top=500):
        props = self.sst(feats)
        props, scores = self.sst.get_segments(props)
        nms_props, nms_scores = self.sst.nms_detections(props, scores)

        return props[:top]

    def features(self, video, interval):
        return self.c3d([video], [[interval]], [self.dataset.duration[video]])


class Demo_2:

    def __init__(self, dataset, config):

        self.index   = [] # will contain [{ interval, caption, similarity }]
        self.config  = config
        self.dataset = dataset

        self.c3d    = actionet.ANC_C3D(config)
        self.sst    = sst.SST(config)
        self.softie = softcap.SoftCap(dataset.vocab_size, config.features, config.caption)
        self.skip   = skipthought.SkipThought(dataset, config)

        self.sst.load_state_dict(T.load(config.paths["sst_weights"]))
        self.softie.load_state_dict(T.load(config.paths["softcap_weights"]))

        self.sst.eval()
        self.softie.eval()

        self.sst.cuda()
        self.softie.cuda()

    def run(self, video, query):
        #ntervals = [[0, 5.17], [5.17, 30.29], [30.29, 107.87], [72.41, 87.19], [88.66, 107.13], [108.61, 114.52], [114.52, 137.43], [138.17, 147.77]]
        intervals = self.dataset.gt_intervals[video]
        for interval in intervals:
            feats = self.features(video, interval)[0]
            feat_mask = np.zeros(self.config.features["n_time_steps"])
            feat_mask[:feats.shape[1]] = 1

            caption = self.captions(feats, [feat_mask])
            similarity = self.match_query(caption, query)
            
            print("[DEMO] Similarity: {}".format(similarity))
            print("[DEMO] Caption: {}".format(caption))

            self.index += [{"caption": caption, "interval": interval, "score": similarity}]

        self.index = sorted(self.index, key=lambda k: k['score'])[::-1]
        return json.dumps(self.index)

    def match_query(self, caption, query):
        similarity = self.skip(caption, query)
        return similarity

    def captions(self, feats, feat_mask):
        features  = Variable(T.FloatTensor(feats)).cuda()
        feat_mask = Variable(T.FloatTensor(feat_mask)).cuda()
        gen_words = self.softie.evaluate(self.config.caption["batch_size"], features, feat_mask)
        gen_words = gen_words[1][0]
        eos = np.where(gen_words==2)[0]
        eos = eos[0] if len(eos) != 0 else len(gen_words)
        caption = ' '.join(self.dataset.idx2word[gen] for gen in gen_words[:eos])
        return caption

    def proposals(self, feats, top=500):
        props = self.sst(feats)
        props, scores = self.sst.get_segments(props)
        nms_props, nms_scores = self.sst.nms_detections(props, scores)
        return props[:top]

    def features(self, video, interval):
        return self.c3d([video], [[interval]], [self.dataset.duration[video]])

class Demo_3:

    def __init__(self, dataset, config):
        self.stopwords = stopwords.words('english')

        self.index   = [] # will contain [{ interval, caption, similarity }]
        self.config  = config
        self.skip   = skipthought.SkipThought(dataset, config)

        self.preprocessed_vids = json.load(open("/home/balkhamissi/projects/fmn2/results/xc-s2vt-4-final.json","r"))
        self.preprocessed_vids = self.preprocessed_vids["results"]

        self.forrest_gump = json.load(open("/home/balkhamissi/projects/fmn2/results/xc-softcap-0009.json","r"))
        self.forrest_gump = self.forrest_gump["results"]

        self.titanic = json.load(open("/home/balkhamissi/projects/fmn2/results/xc-softcap-1010.json","r"))
        self.titanic = self.titanic["results"]

    def time2sec(self, time_str):
        return sum(x * int(t) for x, t in zip([3600, 60, 1, 0.001], time_str.split(".")))

    def convert_to_timestamp(self, file_name):
        # example: 0009_Forrest_Gump_00.57.31.159-00.57.34.563.avi
        v_id = file_name[:file_name.rfind('.')]
        time_range = v_id[v_id.rfind('_')+1:]
        start = time_range[:time_range.find('-')]
        end   = time_range[time_range.find('-')+1:]
        start = self.time2sec(start)
        end   = self.time2sec(end)
        return [start, end]

    def run(self, videos, query : str):
        self.index = []
        
        def pre0009(vid_id, movie_ds):
            caps = []
            results = []
            w2cap = {}
            ints = []

            qw = list(set(query.split(' ')) - set(self.stopwords))
            for seg in movie_ds:
                interval = self.convert_to_timestamp(seg)
                caption  = movie_ds[seg][0]["sentence"]
                caps.append({'interval': interval, 'caption': caption})
                for x in caption.split(' '):
                    if x not in w2cap:
                        w2cap[x] = []
                    if interval not in ints:
                        w2cap[x].append({'interval': interval, 'caption': caption})
                        ints.append(interval)

            cands = []
            for w in qw:
                # TODO! check for synonyms
                if w in w2cap:
                    cands.extend(w2cap[w])
            
            for c in cands:
                # for sm in c:
                caption = c['caption']
                word_sim = len(set(caption.split(' ')).intersection(qw))/len(qw)
                vectory_sim = self.match_query(caption, query)
                score = word_sim*0.4 + vectory_sim*0.6
                results += [{"vid_id": vid_id, "caption": caption, "interval": c['interval'], "score": score}]
            return results

        for vid in videos:
            if vid == "0009":
                res = pre0009(vid, self.forrest_gump)
                self.index = res
                continue
            elif vid == "1010":
                res = pre0009(vid, self.titanic)
                self.index = res
                continue


                for seg in self.forrest_gump:
                    interval = self.convert_to_timestamp(seg)
                    caption  = self.forrest_gump[seg][0]["sentence"]
                    similarity = self.match_query(caption, query)
                    
                    self.index += [{"vid_id": vid,"caption": caption, "interval": interval, "score": similarity}]
            else:
                for idx in range(len(self.preprocessed_vids[vid][:20])):
                    interval = self.preprocessed_vids[vid][idx]["timestamp"]
                    caption =  self.preprocessed_vids[vid][idx]["sentence"]
                    similarity = self.match_query(caption, query)
                
                    self.index += [{"vid_id": vid[2:],"caption": caption, "interval": interval, "score": similarity}]

        self.index = sorted(self.index, key=lambda k: k['score'])[::-1]
        self.index = self.index[:10]

        return json.dumps(self.index)

    def match_query(self, caption, query):
        similarity = self.skip(caption, query)
        return similarity

