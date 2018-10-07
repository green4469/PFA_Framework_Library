import PFA
import numpy as np
import tester

def pfa2input(pfa, file_name):
    num_states = pfa.nbS
    alphabet_size = len(pfa.alphabet)
    num_transitions = 0
    for key in pfa.transitions.keys():
        num_transitions += np.count_nonzero(pfa.transitions[key])

    f = open(file_name, "w")
    print(num_states, alphabet_size, num_transitions, file=f)
    for i in range(num_states):
        print(pfa.initial[i], pfa.final[i], file = f)
    for key in pfa.transitions.keys():
        for i in range(num_states):
            for j in range(num_states):
                if pfa.transitions[key][i,j] > 0:
                    print(i, key, j, pfa.transitions[key][i,j], file = f)
    f.close()

if __name__ == "__main__":
    at = tester.parser("./inputs/input4_new.txt")
    pfa2input(at, "input4.txt")