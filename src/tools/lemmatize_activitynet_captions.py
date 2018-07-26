from lemmatizer import *
import sys
import json
import os

def lemmatize_activitynet_captions(data):
    lemmatized_data = data
    for video in data:
        for index, sentence in enumerate(data[video]["sentences"]):
            lemmatized_data[video]["sentences"][index] = get_lemmatized(sentence)
    return lemmatized_data

in_file = open(sys.argv[1],"r")
data = json.load(in_file)
in_file.close()
out_file = open(os.path.splitext(os.path.basename(sys.argv[1]))[0]+"_l.json","w")
out_file.write(json.dumps(lemmatize_activitynet_captions(data)))
out_file.close()