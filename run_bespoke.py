import argparse, sys, os
from modules import bespoke_core, common

def check_file_srcs(args):
    stat = True
    if not os.path.isfile(args.nw_src):
        print "Error: nw_src file not found"
        stat = False
    if not os.path.isfile(args.tr_src):
        print "Error: tr_src file not found"
        stat = False
    if args.ls!=None:
        if not os.path.isfile(args.ls) :
            print "Error: label src file not found"
            stat = False
    if args.eval_src!=None:
        if not os.path.isfile(args.eval_src):
            print "Error: eval_src file not found"
            stat = False
    return stat

def get_parser():
    parser = argparse.ArgumentParser(description='Description: Script to run Bespoke on a given graph \
                                     and known(training) communities file. Please refer to the README \
                                     for details about each argument.')
    parser.add_argument('nw_src', help='Graph file. One edge per line.')
    parser.add_argument('tr_src', help='File containing training comms.')
    parser.add_argument('num_find', type=int, help='Number communities to extract.')
    parser.add_argument('ls', help='File containing node labels.')
    parser.add_argument('op', help='Write discovered communities to this file.')

    parser.add_argument('--np', type=int, help='Number of subgraph patterns to extract. Default=5', default=5)
    parser.add_argument('--eval_src', help='File containing comms. to evaluate against.')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    try:
        a = parser.parse_args()
    except:
        exit()
        
    print "\n####\tInput arguments\t###"
    for k in vars(a):
        print k, "-->",getattr(a,k)
    print "####\n"

    stat = check_file_srcs(a)
    if stat == True:
        print "\n###\t Beginning Bespoke\t###"
        ret = bespoke_core.main(a.nw_src, a.tr_src, a.ls, a.num_find, a.np)
        if ret != None:
            found_comms, KM_obj, tot_time, train_time = ret
            print "total_time(s):",tot_time
            print "training_time(s):",train_time
            print "Writing extracted communities to file:",a.op,"...",
            common.write_comms(a.op, found_comms)
            print "done"
            print "###\t End of Bespoke\t###"


            if a.eval_src!=None:
                test_comms = common.load_comms(a.eval_src)
                if len(test_comms)==0:
                    print "No test comms of size >3 found. Cannot evaluate found communities."
                    exit()
                print "\nScoring extracted communities (may take a lot of time)..."
                sys.stdout.flush()
                score = common.combined(found_comms, test_comms)
                print "F1 Score:",score[0]
