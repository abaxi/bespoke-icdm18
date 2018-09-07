import sys, numpy, os, random, time
import common, train
from sklearn.cluster import KMeans

def get_seed(gt_size, seed_dict):
    degree_start = gt_size-1
    degree_end = degree_start
    eps = 5
    while True:
        if degree_end in seed_dict:
            ordered_list = seed_dict[degree_end].ordered_list
            if len(ordered_list) != 0:
                seed, score = ordered_list.pop(0)
                return seed, degree_end
        degree_end+=1
        if degree_end - degree_start>eps:
            return None, degree_end

def get_supports(size_dist_per_group):
    d = {}
    tot = float(sum(map(len, size_dist_per_group.values())))
    for g_id in size_dist_per_group:
        supp = len(size_dist_per_group[g_id])
        d[g_id] = supp/tot
    return d

def pick_pattern(pattern_supports):
    r = random.random()
    s = 0
    for patt_id in pattern_supports:
        s+=pattern_supports[patt_id]
        if s > r:
            return patt_id
    return patt_id

def pick_size(size_dist):
    return random.sample(size_dist,1)[0]

def get_comms(nw, num_find, seeds_by_group, size_dist_per_group, KM_obj, node_labels, unique_seeds, rep_th=2):
    found_size_pat_dist = {}
    comms_list = []
    seen = {}
    pattern_supports = get_supports(size_dist_per_group)
    while len(comms_list) < num_find:
        group_id = pick_pattern(pattern_supports)
        size_dist = size_dist_per_group[group_id]
        size = pick_size(size_dist)
        seed_dict = seeds_by_group[group_id]
        seed, new_deg= get_seed(size, seed_dict)
        if unique_seeds:
            while seen.get(seed, 0) > rep_th:
                seed, new_deg = get_seed(size, seed_dict)
        comm = nw.rand_subgraph_nodes(size, seed)
        if len(comm)>=common.MIN_COM_SIZE:
            comms_list.append(comm)
            if unique_seeds:
                for n in comm:
                    if n not in seen:
                        seen[n]=0
                    seen[n]+=1
    return comms_list

def main(nw_src, gt_src, node_label_src, num_find, nclus, verbose=True, unique_seeds=False):
    start = time.time()
    gts = common.load_comms(gt_src)
    if len(gts) == 0:
        print "Error! No training communities of size >3 found. Cannot run Bespoke."
        return None
    if len(gts) < nclus:
        print "Error! Too few (<#patterns,",nclus,") training communities of size >3 found. Cannot run Bespoke."
        return None
    node_labels = common.load_labels(node_label_src)
    nw = common.load_SimpleNW_graph(nw_src)

    if len(nw.nodes()) != len(node_labels.keys()):
        print "Error! Number of labeled nodes does not match number of nodes in the graph. #NW nodes=", len(nw.nodes()),"#labeled nodes=", len(node_labels.keys())
        return None
    if verbose:
        print "Training..."
        sys.stdout.flush()
    ret_list = train.train(nw, gts, node_labels, nclus)

    if verbose:
        print "Training complete...\nBeginning extraction..."
        sys.stdout.flush()
    KM_obj, size_dist_per_group, seed_info_per_group = ret_list
    end_train = time.time()
    comms = get_comms(nw, num_find, seed_info_per_group, size_dist_per_group, KM_obj, node_labels, unique_seeds)
    end = time.time()
    tot_time, train_time = round(end-start,2),round(end_train-start,2)

    if verbose:
        print "Extraction complete."
        sys.stdout.flush()
    return comms, KM_obj, tot_time, train_time
