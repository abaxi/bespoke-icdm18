import sys, operator, os
import SimpleNW
import numpy as np
MIN_COM_SIZE = 2

def load_SimpleNW_graph(srcFilePath, delim="\t"):
    nw = SimpleNW.SimpleNW()
    nw.load_ugraph_from_file(srcFilePath,delim)
    return nw

def load_comms(src, delim="\t", threshold=MIN_COM_SIZE):
    f = open(src,'r')
    gts = {}
    gtID = 1
    try:
        for line in f:
            if line[0] == "#":
                continue
            if line.strip() == "":
                continue
            s = set(map(int, line.strip().split(delim)))
            if len(s)<threshold:
                continue
            gts[gtID] = s
            gtID+=1
    except:
        print "couldn't process line: ",line
        f.close()
        exit()
    return gts.values()

def write_comms(name, comms_list):
    f = open(name, 'w')
    for comm in comms_list:
        o = "\t".join(map(str,comm))+"\n"
        f.write(o)
    f.close()

def load_labels(src, delim="\t"):
    f = open(src,'r')
    d = {}
    for line in f:
        line = line.strip()
        if line == "":
            continue
        if line[0] == "-":
            continue
        node, role = map(int,line.split(delim))
        d[node] = role
    f.close()
    return d

def load_node_features(src,delim="\t"):
    f = open(src,'r')
    d = {}
    for line in f:
        if line[0] == "#":
            continue
        parts = line.strip().split(delim)
        nodeID = int(parts[0])
        features = map(float, parts[1:])
        d[nodeID] = features
    f.close()
    return d

def make_dict(keys, values):
    d = {}
    for i in range(len(keys)):
        d[keys[i]] = values[i]
    return d

def myCounter(l):
    s = {}
    final_s = set()
    for n in l:
        if n not in s:
            s[n]=0
        s[n]+=1
    return s

def normalize_counts(d):
    d_normed = {}
    tot = float(sum(d.values()))
    for k in d.keys():
        d_normed[k] = d[k]/tot
    return d_normed

def get_canonical(labeled_tuple):
    if labeled_tuple[0]<labeled_tuple[-1]:
        labeled_tuple = labeled_tuple[::-1]
    return tuple(labeled_tuple)

##########################
####Code to compute F1_score
##########################


def combined_helper(found_sets, GT_sets):
    d1 = {} #best match for an extracted community
    d2 = {} #best match for a known community

    for i in range(len(GT_sets)):
        gt = GT_sets[i]
        f_max = 0

        for j in range(len(found_sets)):
            f = found_sets[j]

            common_elements = gt.intersection(f)
            if len(common_elements) == 0:
                temp = 0
            else:
                temp = F_score_helper(gt, f, common_elements)

            f_max = max(f_max,temp)

            d1[j] = max(d1.get(j,0),temp)

        d2[i] = f_max

    return d1, d2

def combined(found_sets, GT_sets, verbose=False):
    d1,d2 = combined_helper(found_sets, GT_sets)

    if d1 == None:
        return [0]*6

    vals1 = sum(d1.values())/len(d1)
    vals2 = sum(d2.values())/len(d2)
    f_score = vals1+vals2
    f_score/=2
    f_score = round(f_score,4)
    vals1 = round(vals1,4)
    vals2 = round(vals2,4)

    return f_score, vals1, vals2

def F_score_helper(GT, found, common_elements):
    len_common = len(common_elements)
    precision = float(len_common)/len(found)
    if precision == 0:
        return 0

    recall = float(len_common)/len(GT)
    if recall == 0:
        return 0

    return (2*precision*recall)/(precision+recall)
##########################
#### end
##########################
