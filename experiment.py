from common_header import *
import lev
import itertools

import PFA
import PFA_utils


# Read input file
# Parse input file into DPFA instance

# sys.argv[1] : DPFA input directory name
# sys.argv[2] : string file name
# sys.argv[3] : 'bf' for brute force and 'dp' for dynamic progarmming
# sys.argv[4] : k range

def str2list(s):
    return [int(k) for k in s.split(',')]




def main(args):
    k_range = args.k_range
    k_range[1] += 1
    k_range = range(*k_range)
    nbS = args.nbS
    nbL = args.nbL
    max_n = args.max_n
    iters = args.iters
    result_path = args.result_path
    if os.path.exists(result_path+'.csv'):
        print(result_path,'exists')
        sys.exit()
    with open(result_path+'.csv', 'w+') as f:
        f.write('k,n,nbS,nbL,RT_DP,RT_intersect,input_string,mps_DP,mps_intersect,intersect_time,normalize_time\n')

    # 1. generate a DPFA and string w \in \Sigma s.t. |w| = n
    dpfa = PFA_utils.DPFA_generator(nbS, nbL)
    print("DPFA generated")
    print("loading to GPU...")
    dpfa.use_cuda()
    PFA_utils.pfa2input(dpfa, result_path+".dpfa")
    sigma = [str(chr(ord('a')+i)) for i in range(nbL)]
    for n in range(1, max_n+1):
        for k in k_range:
            if k > n:
                continue
            for i in range(iters):
                #w = dpfa.generate()
                #n = np.random.randint(k, max_n+1)
                w = ''.join(np.random.choice(sigma, n))
                prob = dpfa.parse(w)
                """
                while len(w) == 0 or prob ==0.0:
                    #w = dpfa.generate()
                    n = np.random.randint(1, max_n+1)
                    w = ''.join(np.random.choice(sigma, n))
                    prob = dpfa.parse(w)
                n = len(w)
                if k > n:
                    k = np.random.randint(1, n+1)
                """
                RT = dict()
                mps = dict()
                for algorithm in ['dp', 'intersect']:
                    print("[{}] k: {}, n: {}, nbS: {}, nbL: {}".format(algorithm, k, n, nbS, nbL))
                    print('given string',w,'with prob',prob)
                    # 2. run MPS
                    start_time = time.time()
                    if 'intersect' in algorithm.lower():
                        print('hamming automaton...')
                        dfa = PFA_utils.DFA_constructor(w, k, sigma)
                        print('nbS of the hamming automaton: {}'.format(dfa.nbS))
                        print('intersecting...')
                        intersect_start_time = time.time()
                        sub_dpfa = dpfa.intersect_with_DFA(dfa)
                        sub_dpfa = PFA.PFA(sub_dpfa.nbL, sub_dpfa.nbS, sub_dpfa.initial, sub_dpfa.final, sub_dpfa.transitions)
                        intersect_time = time.time() - intersect_start_time
                        print('nbS of intersected DPFA: {}'.format(sub_dpfa.nbS))
                        normalize_start_time = time.time()
                        if sub_dpfa.nbS > 0:
                            print('normalizing...')
                            normalized_dpfa = PFA_utils.normalizer(sub_dpfa)
                            normalize_time = time.time() - normalize_start_time
                            print('MPS...')
                            mps[algorithm] = normalized_dpfa.MPS()
                        else:
                            mps[algorithm] = None
                            normalize_time = time.time() - normalize_start_time
                    elif 'dp' in algorithm.lower():
                        mps[algorithm] = dpfa.k_MPS(w, k)
                    elif 'bf' in algorithm.lower():
                        mps[algorithm] = dpfa.k_MPS_bf(w, k)
                    else:
                        raise NotImplementedError
                    print('done!')
                    end_time = time.time()
                    RT[algorithm] = end_time - start_time
                    print('time elapsed: {:.4f}s'.format(RT[algorithm]))
                # 3. record RT
                if sub_dpfa.parse(mps['dp']) == 0.:
                    mps['dp'] = None
                print('[RESULTS] {}: {}, {}: {}'.format('dp', mps['dp'], 'intersect', mps['intersect']))
                with open(result_path+'.csv', 'a') as f:
                    f.write('{},{},{},{},{:.5f},{:.5f},{},{},{},{},{}\n'.format(
                            k, n, nbS, nbL, RT['dp'], RT['intersect'], w,mps['dp'],mps['intersect'],intersect_time,normalize_time))



if __name__ == "__main__":
    parser = argparse.ArgumentParser('experiment for the comparision of DP based algorithm and intersection based algorithm')

    parser.add_argument('--k_range', type=str2list, help='distance limit k')
    parser.add_argument('--nbS', type=int, help='number of states of PFA')
    parser.add_argument('--nbL', type=int, help='set size of alphabets')
    parser.add_argument('--iters', type=int, help='# of iters per string')
    parser.add_argument('--max_n', type=int, help='maximum size of n')
    parser.add_argument('--result_path', type=str, help='name of the result')

    args = parser.parse_args()
    main(args)


"""
###for previous experiment###

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


