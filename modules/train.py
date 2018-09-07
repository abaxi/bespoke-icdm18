import sys, topN, numpy
import common
from sklearn.cluster import KMeans

def get_ordered_canonical_label_pairs(num_labels):
    seen = set()
    ordered = []
    for i in range(num_labels):
        for j in range(num_labels):
            pair = common.get_canonical((i,j))
            if pair in seen:
                continue
            ordered.append(pair)
            seen.add(pair)
    return ordered

def get_node_score_degree(nw, node, biases, node_labels):
    neigh = nw.neighbors(node)
    labels = map(node_labels.get, neigh)
    labels = common.myCounter(labels)
    root_l = node_labels[node]
    score = 0
    for l in labels:
        pair = common.get_canonical((root_l, l))
        bias = biases.get(pair,[0,0])[0]  ###if bias is never seen before, it is 0
        score += labels[l]*bias
    return score, len(neigh)

def get_node_order_by_degree(nw, biases, node_labels):
    nodes = nw.nodes()
    d = {}
    pass1_scores = {}
    for node in nodes:
        score, degree = get_node_score_degree(nw, node, biases, node_labels)
        score = score/degree
        pass1_scores[node] = score

    for node in nodes:
        degree = nw.degree(node)
        score = get_weighted_avg(nw, node, pass1_scores)
        if degree not in d:
            ordered_list = topN.topN(len(nodes))
            d[degree] = ordered_list
        score = round(score, 2)
        if score < 0.2:
            continue
        d[degree].insert_update(node, score)
    return d

def get_weighted_avg(nw, node, scores):
    neighs = nw.neighbors(node)
    s = 0.0
    all_deg = 0.0
    for n in neighs:
        n_score = scores[n]
        n_deg = nw.degree(n)
        s+=(n_score*n_deg)
        all_deg+=n_deg
    s/=all_deg
    s=s+scores[node]
    return s

def get_biases(features, num_labels, node_labels):
    ordered_pairs = get_ordered_canonical_label_pairs(num_labels)
    x = numpy.array(features)
    means = numpy.mean(x,0)
    devs = numpy.std(x,0)
    d = {}
    for i in range(len(ordered_pairs)):
        d[ordered_pairs[i]] = (means[i], devs[i])
    return d

def get_seed_infos(nw, grouped_features, node_labels, num_labels, nclus):
    seeds_by_group = {}
    biases_by_group = {}
    for group_id in grouped_features.keys():
        biases = get_biases(grouped_features[group_id], num_labels, node_labels)
        biases_by_group[group_id] = biases
        node_order_by_degree = get_node_order_by_degree(nw, biases, node_labels)
        seeds_by_group[group_id] = node_order_by_degree
    return seeds_by_group, biases_by_group

def make_groups(ids, gts, gt_features):
    size_dist_per_group = {}
    grouped_features = {}
    for i in range(len(gts)):
        size_dist_per_group.setdefault(ids[i], []).append(len(gts[i]))
        grouped_features.setdefault(ids[i], []).append(gt_features[i])
    return size_dist_per_group, grouped_features

def get_features_helper(nw, comm, node_labels, num_labels):
    s = nw.subgraph(comm)
    e_list = s.edges()
    dist = []
    for e in e_list:
        e = map(node_labels.get, e)
        e = common.get_canonical(e)
        dist.append(e)
    dist = common.normalize_counts(common.myCounter(dist))
    feature_vector = []
    ordered_label_pairs = get_ordered_canonical_label_pairs(num_labels)
    for label_pair in ordered_label_pairs:
        r = round(dist.get(label_pair,0), 4)
        feature_vector.append(r)
    return feature_vector

def get_features(nw, comms, node_labels, num_labels):
    comms_features = []
    for comm in comms:
        feature_vector = get_features_helper(nw, comm, node_labels, num_labels)
        comms_features.append(feature_vector)
    return comms_features

def train(nw, gts, node_labels, nclus, Simple=False):
    num_labels = len(set(node_labels.values()))
    gt_features = get_features(nw, gts, node_labels, num_labels)
    KM_obj = KMeans(nclus)
    gt_labels = KM_obj.fit_predict(gt_features)
    size_dist_per_group, grouped_features = make_groups(gt_labels, gts, gt_features)
    if Simple:
        return KM_obj, size_dist_per_group
    seed_info_per_group, biases_by_group = get_seed_infos(nw, grouped_features, node_labels, num_labels, nclus)
    return KM_obj, size_dist_per_group, seed_info_per_group
