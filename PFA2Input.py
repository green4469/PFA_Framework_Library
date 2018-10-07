import PFA
import numpy as np
import tester

def pfa2input(pfa):
    num_states = pfa.nbS
    alphabet_size = len(pfa.alphabet)
    num_transitions = 0
    for key in pfa.transitions.keys():
        num_transitions += np.count_nonzero(pfa.transitions[key])

    f = open("input.txt", "w")
    print(num_states, alphabet_size, num_transitions, file=f)
    #f.write(num_states, alphabet_size, num_transitions)
    for i in range(num_states):
        print(pfa.initial[i], pfa.final[i], file = f)
        #f.write(pfa.initial[i], pfa.finail[i])
    for key in pfa.transitions.keys():
        for i in range(num_states):
            for j in range(num_states):
                if pfa.transitions[key][i,j] > 0:
                    print(i, key, j, pfa.transitions[key][i,j], file = f)
                    #f.write(i, key, j, pfa.transitions[key][i,j])
    f.close()

if __name__ == "__main__":
    at = tester.parser("./inputs/input_new.txt")
    pfa2input(at)