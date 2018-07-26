import sys
import pandas
import json

def time2sec(time_str):
    return sum(x * int(t) for x, t in zip([3600, 60, 1, 0.001], time_str.split(".")))

def convert(v_id):
    movie_id = v_id[:4]
    time_range = v_id[v_id.rfind('_')+1:]
    start = time_range[:time_range.find('-')]
    end   = time_range[time_range.find('-')+1:]
    start = time2sec(start)
    end   = time2sec(end)

    return f"{movie_id}_{start}-{end}"

def main(csv_path):
    # vid_files = []
    manifest = {}
    csv = pandas.read_csv(csv_path, header=None, sep='\t')
    ids = csv.iloc[:, [0]].values.tolist()
    csv = csv.set_index(0)
    print()
    l = len(ids)
    for i, [v_id] in enumerate(ids):
        # v_id = vf[:vf.rfind('.')]
        v = csv.iloc[i]
        #v_id2 = convert(v_id)
        # interval = [v[0], v[1]]
        sentence = v.iloc[-1]
        manifest[v_id+".avi"] = sentence
        # manifest[v_id[:4]]['timestamps'].append(interval)
        # manifest[v_id[:4]]['sentences'].append(sentence)
        print(f'\r{i:d}/{l:d} {i/l*100:3.1f}%', end='')

    print('\n*** Writing')
    with open('lsmdc_val-1.json', 'w') as fp:
        json.dump(manifest, fp)


if __name__ == '__main__':
    main(sys.argv[1])
    print('*** Done')
