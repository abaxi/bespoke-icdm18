import numpy, sys, time, os
from modules import common
from sklearn.cluster import KMeans
from scipy.stats.mstats import zscore
import argparse

##############################################
#####       Node feature extraction
#####
##############################################

def get_jaccard_dict(graph):
    edges = set(graph.edges())

    nodes = list(graph.nodes())
    num_edges = len(edges)
    perc_1 = num_edges/100
    jaccs = {}
    d_neigh = {}

    for e in edges:
        a,b = e
        if a in d_neigh:
            neigh_a = d_neigh[a]
        else:
            neigh_a = set(graph.neighbors(a))
            d_neigh[a] = neigh_a

        if b in d_neigh:
            neigh_b = d_neigh[b]
        else:
            neigh_b = set(graph.neighbors(b))
            d_neigh[b] = neigh_b

        if a not in jaccs:
            jaccs[a] = []

        if b not in jaccs:
            jaccs[b] = []

        common_nodes = neigh_b.intersection(neigh_a)
        common_nodes = common_nodes.difference([a,b])
        num_common = len(common_nodes)
        union = max(float(len(neigh_b.union(neigh_a))), 1)
        jacc = num_common/union

        jaccs[a].append(jacc)
        jaccs[b].append(jacc)

    d = {}
    for node in jaccs.keys():
        v = jaccs[node]
        v = numpy.percentile(v, [0,25,50,75,100])
        v = numpy.round(v,3)
        s = map(str, v)
        s = "\t".join(s)
        d[node] = s
    return d

def write_node_feature(data_dict, outname):
    f = open(outname,'w')
    data = []
    for node in data_dict.keys():
        jacc_list = data_dict[node]
        s = str(node)+"\t"+jacc_list+"\n"
        data.append(s)

        if len(data) > 10000:
            for d in data:
                f.write(d)
            data = []

    if len(data) > 0:
        for d in data:
            f.write(d)
        data = []

    f.close()

def run_node_feature(graph_src, verbose=False):
    print "processing ",graph_src
    src = graph_src

    start = time.time()
    g = common.load_SimpleNW_graph(src)
    if verbose:
        print "graph loaded in ", round(time.time()-start,2), "seconds"
        print "-----------------"
        sys.stdout.flush()

    start = time.time()
    jaccard_dict = get_jaccard_dict(g)
    if verbose:
        print "jaccard percentiles computed in ", round(time.time()-start,2), "seconds"
        sys.stdout.flush()

    start = time.time()
    out_name= src[src.rfind(os.sep)+1:]
    out_name = out_name.replace(".txt","_jacc_perc_NodeFeatures.txt")
    if verbose:
        print "features written to file:", out_name
    write_node_feature(jaccard_dict, out_name)
    if verbose:
        print "Jaccard node features extracted/written in ", round(time.time()-start,2), "seconds"
        print "-----------------"
        sys.stdout.flush()
    return out_name



##############################################
#####       Node clustering/labeling
#####
##############################################
def cluster_nodes(nodes, features, args_list):
    K = args_list[0]
    c = KMeans(K)
    c.fit(features)
    labels = c.labels_

    label_dict = common.make_dict(nodes, labels)
    return label_dict

def write_roles(d_roles,name, verbose):
    f = open(name,'w')
    for key in d_roles:
        s = str(key)+"\t"+str(d_roles[key])+"\n"
        f.write(s)
    f.close()
    if verbose:
        print "Node labels written to file:", name
        sys.stdout.flush()

def process_graph(src_node_features, K, verbose=False):
    nodes_features = common.load_node_features(src_node_features, delim="\t")
    if verbose:
        print "node features loaded"
        sys.stdout.flush()

    all_nodes = nodes_features.keys()
    data = numpy.array(nodes_features.values())
    data = zscore(data, 0)
    data = numpy.nan_to_num(data)
    label_dict = cluster_nodes(all_nodes, data, [K])
    if label_dict == None:
        return None
    if verbose:
        print "nodes clustered"
        sys.stdout.flush()
    return label_dict


def get_parser():
    parser = argparse.ArgumentParser(description='Description: Script to run node labeling on a given graph. \
                                     Please refer to the readme for details about each argument.')
    parser.add_argument('nw_src', help='Graph file. One edge per line.')
    parser.add_argument('n_labels', type=int, help='Number of labels. Node is assigned 1 of N labels.', default=4)
    parser.add_argument('op', help='Write node labels to this file.')
    parser.add_argument('--verbose', help='enable verbosity.', action="store_true")
    return parser

################################################
####        Main hook
####
################################################
if __name__ == '__main__':
    parser = get_parser()
    try:
        a = parser.parse_args()
    except:
        exit()
    
    graph_src = a.nw_src
    K = a.n_labels
    name = a.op
    verbose = a.verbose

    print "\n###\tGenerating node labels\t###"
    feature_src = run_node_feature(a.nw_src, a.verbose)
    label_dict = process_graph(feature_src, a.n_labels, a.verbose)
    write_roles(label_dict, a.op, a.verbose)
