from common_header import *
import lev

import PFA
import PFA_utils
import argparse
import random


# Read input file

# Parse input file into DPFA instance

# sys.argv[1] : DPFA input directory name
# sys.argv[2] : string file name
# sys.argv[3] : 'bf' for brute force and 'dp' for dynamic progarmming
# sys.argv[4] : k range

def str2list(s):
    return [int(k) for k in s.split(',')]


    

def main(args):
    algorithm = args.algorithm.lower()
    k_range = args.k_range
    n_range = args.n_range
    nbS_range = args.nbS_range
    nbL_range = args.nbL_range
    iters = args.iters
    result_path = args.result_path
    for i in range(iters):
        k = random.randrange(*k_range)
        n = random.randrange(*n_range)
        nbS = random.randrange(*nbS_range)
        nbL = random.randrange(*nbL_range)
        """
        TO DO:
            1. generate string w given n
            2. generate dpfa A given nbS and nbL
            3. run the given algorithm
            4. save the result with type_of_algorithm, k, n, nbS, nbL, nbT, and running time to the result path
        """


if __name__ == "__main__":
    parser = argparse.ArgumentParser('experiment for the comparision of DP based algorithm and intersection based algorithm')

    parser.add_argument('--algorithm', type=str, help='type of MPS algorithm: DP and intersection')
    parser.add_argument('--k_range', type=str2list, help='distance limit k')
    parser.add_argument('--n_range', type=str2list, help='string length n')
    parser.add_argument('--nbS_range', type=str2list, help='number of states of PFA')
    parser.add_argument('--nbL_range', type=str2list, help='set size of alphabets')
    parser.add_argument('--iters', type=int, help='number of iterations')
    parser.add_argument('--result_path', type=str, help='name of the result')

    args = parser.parse_args()
    main(args)


"""

DPFA_input_files = os.listdir(sys.argv[1])
k_min_max = str2list(sys.argv[4])
k_range = range(k_min_max[0], k_min_max[1]+1)

with open(sys.argv[2], 'r') as f:
    strings = f.readlines()
    strings = [string[:-1] for string in strings]  # remove '\n'
f = open('./test_result_{}.csv'.format(sys.argv[3]), 'w+')
f.write('n,k,rt\n')

for input_file in DPFA_input_files:
    DPFA_instance = PFA_utils.parser('inputs/pfa/'+input_file)
    for k in k_range:
        for string in strings:
            n = len(string)
            if n < k or n < 20:
                continue
            st = time.time()
            if sys.argv[3] == 'bf':
                k_MPS = DPFA_instance.k_MPS_bf(string, k)
            elif sys.argv[3] == 'dp':
                k_MPS = DPFA_instance.k_MPS(string, k)
            elif sys.argv[3] == 'is':
                l = lev.LevinshteinAutomata(string, k)
                sub_dpfa = DPFA_instance.intersect_with_DFA(l)
                sub_dpfa = PFA.PFA(sub_dpfa.nbL, sub_dpfa.nbS, sub_dpfa.initial, sub_dpfa.final, sub_dpfa.transitions)
                dpfa = PFA_utils.normalizer(sub_dpfa)
                k_MPS = dpfa.MPS_sampling()

            et = time.time()

            interval = et-st
            print("n = {}, k = {}, running time of {}: {:.4f}".format(n, k, sys.argv[3], interval))
            f.write('{},{},{:.4f}\n'.format(n, k, interval))

f.close()
"""


