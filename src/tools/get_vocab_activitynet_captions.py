import sys
import json
import os

def get_vocab(data):
    vocab = {}
    for video in data:
        for sentence in data[video]["sentences"]:
           sentence = sentence.split()
           for word in sentence:
                word ="".join(c for c in word if c not in (';', ':', ',', '.', '!', '?'))
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
    return vocab

in_file = open(sys.argv[1],"r")
data = json.load(in_file)
in_file.close()
vocab = get_vocab(data)
print(len(vocab))