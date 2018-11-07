from common_header import *

import PFA
import PFA_utils
import RA


total_start = time.time()

# Generate a DFA that accepts strings only within k-hamming distance from given input string
start = time.time()
dfa = PFA_utils.DFA_constructor('abab', 2, ['a', 'b'])  # w, k, sigma
end = time.time()
print('Finished dfa construction...', end-start)

# Genearte a random DPFA
start = time.time()
dpfa = PFA_utils.DPFA_generator(20, 7)  # nbS, nbL
end = time.time()
print('Finished dpfa genearation...', end-start)

# Intersect above two automata, the resulting automata is a sub-DPFA
start = time.time()
ra = dpfa.intersect_with_DFA(dfa)  # Currently, RA
sub_dpfa = PFA.PFA(ra.nbL, ra.nbS, ra.initial, ra.final, ra.transitions)
end = time.time()
print('Finished dpfa & dfa intersection...', end-start)

# Normalize the sub_dpfa to dpfa
start = time.time()
dpfa = PFA_utils.normalizer(sub_dpfa)
end = time.time()
print('Finished sub-dpfa normalization...', end-start)

# Do exact MPS on dpfa
start = time.time()
k_mps = dpfa.MPS()
end = time.time()
print('Finished MPS...', end-start)

total_end = time.time()

print(total_end - total_start, k_mps)

