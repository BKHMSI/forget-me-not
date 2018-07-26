import json
from autocorrect import spell
from tqdm import tqdm
# val_1 = json.load(open('fmn2/densevid_eval/data/val_1.json'))
# val_2 = json.load(open('fmn2/densevid_eval/data/val_2.json'))

# xc = json.load(open('fmn2/results/xc-softcap-14.json'))

# vids = list(val_1.keys())
# for vid in vids:
#     if vid not in xc["results"]:
#         del val_1[vid]

# vids = list(val_2.keys())
# for vid in vids:
#     if vid not in xc["results"]:
#         del val_2[vid]

# with open(f'fmn2/results/val_1.json', 'w') as outfile:
#     json.dump(val_1, outfile)

# with open(f'fmn2/results/val_2.json', 'w') as outfile:
#     json.dump(val_2, outfile)

train = json.load(open('fmn2/results/train_autocorrect.json'))

train_ids  = list(train.keys())
captions = {}
for vid in train_ids:
    captions[vid] = train[vid]['sentences']

word_counts = {}
excluded_chars = {ord(c): None for c in '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'}
word_count_threshold = 2

for _, caps in captions.items():
    for sent in caps:
        words = sent.lower().translate(excluded_chars).split(" ")
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1

typos = [w for w in word_counts if word_counts[w] < word_count_threshold]
print('[INFO] Correcting %d from %d words' % (len(typos), len(word_counts)))


# corrected = 0
# for vid in tqdm(train_ids):
#     for i, sent in enumerate(train[vid]["sentences"]):
#         sent_so_far = []
#         for word in sent.split(" "):
#             if word in typos:
#                 sent_so_far += [spell(word)]
#                 corrected += (word != sent_so_far[-1])
#             else:
#                 sent_so_far += [word]
#         train[vid]["sentences"][i] = ' '.join(sent_so_far)

# print(f"[INFO] Corrected: {corrected}")


# with open(f'fmn2/results/train_autocorrect.json', 'w') as outfile:
#     json.dump(train, outfile)
