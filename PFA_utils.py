import numpy as np
from numpy.linalg import inv
import math
import copy
import random
import itertools

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

# Generate a random distribution
def sum_to_one(n):
    values = [0.0, 1.0] + [random.random() for _ in range(n-1)]
    values.sort()
    return [values[i+1] - values[i] for i in range(n)]

# Generate a random PFA input file
MAX_nbT = 3
def generator(fname):
    alphabets = 'abcdefghijklmnopqrstuvwxyz'
    alphabets = [str(alpha) for alpha in alphabets]
    nbL = len(alphabets)
    nbS = random.randint(10, 2000)
    nbT = 0

    initial = sum_to_one(nbS)

    final = []
    transitions = []

    alpha_state_comb = list(itertools.product(alphabets, range(nbS)))

    for i in range(nbS):
        T = random.randint(0, nbS*nbL-1)  # the number of outgoing transitions for this state
        T = min(T, MAX_nbT)
        nbT += T
        probs = sum_to_one(T+1)  # the sum of outgoing transitions probabilities + final probability = 1
        final.append(probs[0])
        for j, comb in enumerate(random.sample(alpha_state_comb, T)):
            transitions.append((i, comb[0], comb[1], probs[1+j]))

    with open(fname, 'w') as f:
        f.write("{} {} {}\n".format(nbS, nbL, nbT))

        for i in range(nbS):
            f.write("{} {}\n".format(initial[i], final[i]))

        for tp in transitions:
            f.write("{} {} {} {}\n".format(tp[0], tp[1], tp[2], tp[3]))


# Verify the generated PFA input files
def verifier(fname):
    at = parser(fname)
    if at.probability_cond()[0] and at.terminating_cond()[0] and int(np.sum(at.get_reachable_state_flag())) == 0:
        return True
    else:
        os.remove(fname)
        return False

def pfa2input(pfa, file_name):
    num_states = pfa.nbS
    alphabet_size = len(pfa.alphabet)
    num_transitions = 0
    for key in pfa.transitions.keys():
        num_transitions += np.count_nonzero(pfa.transitions[key])
    f = open(file_name, "w")

    f.write("{} {} {}\n".format(num_states, alphabet_size, num_transitions))
    for i in range(num_states): 
        f.write("{} {}\n".format(pfa.initial[i], pfa.final[i]))
    for key in pfa.transitions.keys():
        for i in range(num_states):
            for j in range(num_states):
                if pfa.transitions[key][i,j] > 0:
                    f.write("{} {} {} {}\n".format(i, key, j, pfa.transitions[key][i,j]))
    f.close()

