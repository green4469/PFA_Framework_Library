from common_header import *

import PFA_utils

# Read input file

# Parse input file into DPFA instance

# sys.argv[1] : DPFA input directory name
# sys.argv[2] : string file name
# sys.argv[3] : 'bf' for brute force and 'dp' for dynamic progarmming
# sys.argv[4] : k range

def str2list(s):
    return [int(k) for k in s.split(',')]


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
            else:
                k_MPS = DPFA_instance.k_MPS(string, k)
            et = time.time()

            interval = et-st
            print("n = {}, k = {}, running time of {}: {:.4f}".format(n, k, sys.argv[3], interval))
            f.write('{},{},{:.4f}\n'.format(n, k, interval))

f.close()


