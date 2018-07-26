import numpy as np
import json

json_dict = {}

with open("/home/balkhamissi/projects/fmn/dataframe/eval_prop_out.out", 'r') as f:
    f.readline()
    for line in f:
        line = line.split(" ")
        line[-1] = line[-1].replace("\n","")

        if line[-1] not in json_dict:
            json_dict[line[-1]] = []
        
        if len(json_dict[line[-1]]) < 1000:
            json_dict[line[-1]].append([float(line[0]),float(line[1])])

#print(len(json_dict[list(json_dict.keys())[2087]]))
json_str = json.dumps(json_dict)
with open("/home/balkhamissi/projects/fmn2/props@1000.json","w") as f:
   f.write(json_str)
