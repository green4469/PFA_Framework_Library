import numpy as np
from numpy.linalg import inv
import math
import copy

from PFA import PFA


# Receive relative numbers e.g., 3/5
def float_catcher(number):
    try:
        return float(number)
    except:
        a, b = map(int, number.split('/'))
        return float(a/b)

# Parse input txt file and create automaton from it.
def parser(fname):
    with open(fname, 'r') as f:
        # Read the first line and make the string into list. Then automatically pack and unpack it to 3 identifiers, nbS, nbL, nbT
        nbS, nbL, nbT = map(int, f.readline().split(' '))  # nbs for the # of states, nbL for the # of alphabets, nbT for the # of transitions. 

        initial = []
        final = []
        transitions = {}
        
        # For each state, read initial & final probabilities
        for _ in range(nbS):
            i, _f = map(float_catcher, f.readline()[:-1].split(' '))  # Read a line except newline character, '\n'
            
            initial.append(i)
            final.append(_f)

        # For each transition, read initial state, alphabet, final state.
        for _ in range(nbT):
            i, c, _f, w = f.readline()[:-1].split(' ')  # Read a line except newline character, '\n'
            i, _f = int(i), int(_f)
            c = str(c)
            w = float_catcher(w)

            try:
                transitions[c][i,_f] = w
            except:
                transitions[c] = np.zeros((nbS,nbS), dtype=np.float64) 
                transitions[c][i,_f] = w 
    
        initial = np.asarray(initial, dtype=np.float64)
        final = np.asarray(final, dtype=np.float64)


        at = PFA(nbL, nbS, initial, final, transitions)
        return at

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


# Main
if __name__ == "__main__":
    import sys

    automaton = parser(sys.argv[1])  # sys.argv[1] for input file name
    test(automaton, sys.argv[2], int(sys.argv[3]))  # sys.argv[2] for string, 3 for k

