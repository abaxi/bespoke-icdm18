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

##########################
####Code to compute NMI
##########################
def NMI_helper(A_sets, B_sets, num_nodes, norm_dict_A, norm_dict_B):
    d = {} #best match for an extracted community
    for i in range(len(A_sets)):
        a = A_sets[i]
        I_max = 0

        for j in range(len(B_sets)):
            b = B_sets[j]
            temp = get_I_A_B(a, b, num_nodes)
            temp = 2*temp/(norm_dict_A[i]+norm_dict_B[j])
            I_max = max(I_max,temp)
        d[i] = I_max
    
    return d

def get_NMI(found_sets, GT_sets):
    nodes = set()
    for f in found_sets:
        nodes.update(f)
    for g in GT_sets:
        nodes.update(g)
    num_nodes = float(len(nodes))
    norm_dict_found = get_H_dict(found_sets, num_nodes)
    norm_dict_gts = get_H_dict(GT_sets, num_nodes)
    d1 = NMI_helper(found_sets, GT_sets, num_nodes, norm_dict_found, norm_dict_gts)
    NMI = np.mean(d1.values())
    return NMI

def get_I_A_B(a, b, N):
    p11 = 1.0*len(a.intersection(b))/N
    p10 = 1.0*len(a.difference(b))/N
    p01 = 1.0*len(b.difference(a))/N
    p00 = 1-len(a.union(b))/N
    
    p1_a = len(a)/N
    p0_a = 1-p1_a
    p1_b = len(b)/N
    p0_b = 1-p1_b
    
    I = 0
    if p11 * p1_a * p1_b != 0:
        I += p11*np.log2(p11/(p1_a*p1_b))
    if p01 * p0_a * p1_b != 0:
        I += p01*np.log2(p01/(p0_a*p1_b))
    if p00 * p0_a * p0_b != 0:
        I += p00*np.log2(p00/(p0_a*p0_b))
    if p10 * p1_a * p0_b != 0:
        I += p10*np.log2(p10/(p1_a*p0_b))

    return round(I,3)
    
def get_H(a, n):
    p1 = float(len(a))/n
    p0 = 1-p1
    h = 0
    if p1 * p0 == 0:
        h = 0
    else:
        h = p0*np.log2(p0)
        h += p1*np.log2(p1)
        h = -1*h
    
    return h

def get_H_dict(c_list, num):
    d = {}
    for i in range(len(c_list)):
        c = c_list[i]
        d[i] = get_H(c, num)
    return d
