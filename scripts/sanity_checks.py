import os
import sys

import h5py as h5
import numpy as np


# frame_num = 100
# video_id  = 'v_O-YKLVm0ciI'
# db_path = "../data/AN/keras_c3d.h5"

# db_feats = h5.File(db_path, 'r')
# vid_ids  = list(db_feats.keys())
# video_id = vid_ids[np.random.randint(len(vid_ids))]
# print("Video: {}".format(video_id))

# model = K.applications.Xception(include_top=True, pooling='avg')

# vid_feats = db_feats[video_id]
# print("Vid Feat Shape: ", vid_feats.shape)

# vid_feats = vid_feats[frame_num//8]

# model = model[-1]

######### Splitting LSMDC into Training/Validation ############

list_path = 'data/LSMDC/lsmdc-list-1.txt'
fs = sorted(open(list_path, 'r').read().split('\n'))

validation = []
training   = []

movies = {}
for clip in fs:
    movie = clip.split('/')[0]
    vid   = clip.split('/')[-1]
    if movie not in movies:
        movies[movie] = []
    movies[movie] += [vid]

for movie in movies:
    mlen = len(movies[movie])
    perc = mlen // 10
    valm = np.array(movies[movie])
    valm = valm[np.random.choice(range(mlen), perc)]
    validation += list(set(valm))
    print(f"{movie}: {mlen} --> {len(list(valm))}")

for clip in fs:
    vid   = clip.split('/')[-1]
    if vid not in validation:
        training += [vid]

print(f"Training: {len(training)} | Validation {len(validation)}")
print("MUST BE TRUE: ", len(training) + len(validation) == len(fs))
validation = np.array(validation)
training = np.array(training)

np.savetxt('lsmdc_val.txt', validation, delimiter='\n', fmt="%s")
np.savetxt('lsmdc_train.txt', training, delimiter='\n', fmt="%s")




