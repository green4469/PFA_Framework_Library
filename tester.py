from common_header import *
import PFA_utils

# Given a PFA, test that PFA's operations.
def test(at, string, k):
    print('{0:^50}'.format("given string: " + string))
    print('{0:^50}'.format("distance k: " + str(k)))
    print('generate a string                : {}'.format(at.generate()))
    print('probability                      : {}'.format(at.parse(string)))
    print('most probable string             : {}'.format(at.MPS()))
    print('prefix_prob                      : {}'.format(at.prefix_prob(string)))
    print('suffix_prob                      : {}'.format(at.suffix_prob(string)))
    print('probability condition            : {}'.format(at.probability_cond()[0]))
    print('terminating condition            : {}'.format(at.terminating_cond()[0]))
    print('bestpath and highest probability : {}'.format(at.viterbi(string)))
    print('k-MPS                            : {}'.format(at.k_MPS(string, k)))
    print('{0:#^50}'.format(''))
    print()


# Generate n random inputs
def generate(n):
    idx = 1
    for _ in range(n):
        fname = 'inputs/input{}.txt'.format(idx)
        PFA_utils.generator(fname)
        print("Created an input file")
        if PFA_utils.verifier(fname):
            print("Verified #{} input.".format(idx))
            idx += 1
        else:
            print("Failed to verify")

def DPFAgenerate(n):
    idx = 1
    for _ in range(n):
        fname = 'inputs/input{}.txt'.format(idx)
        PFA_utils.DPFAgenerator(fname)
        print("Created an input file")
        if PFA_utils.verifier(fname):
            print("Verified #{} input.".format(idx))
            idx += 1
        else:
            print("Failed to verify")
    

# Main

if __name__ == "__main__":

    """
    automaton = PFA_utils.parser(sys.argv[1])  # sys.argv[1] for input file name

    test(automaton, sys.argv[2], int(sys.argv[3]))  # sys.argv[2] for string, 3 for k
    """
    #generate(int(sys.argv[1]))
    #DPFAgenerate(int(sys.argv[1]))
    #PFA_utils.pfa2input(PFA_utils.normalizer(PFA_utils.parser('inputs/input1.txt')), 'new.txt')


    # DFA, DPFA -> sub-DPFA -> DPFA
    

