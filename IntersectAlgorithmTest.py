from common_header import *

import PFA
import PFA_utils
import RA


total_start = time.time()

# Generate a DFA that accepts strings only within k-hamming distance from given input string
start = time.time()
dfa = PFA_utils.DFA_constructor('ba', 2, ['a', 'b'])  # w, k, sigma
end = time.time()
print('Finished dfa construction...', end-start)

# Genearte a random DPFA
start = time.time()
dpfa = PFA_utils.DPFA_generator(2, 2)  # nbS, nbL
end = time.time()
print('Finished dpfa genearation...', end-start)

# Intersect above two automata, the resulting automata is a sub-DPFA
start = time.time()
ra = dpfa.intersect_with_DFA(dfa)  # Currently, RA
sub_dpfa = PFA.PFA(ra.nbL, ra.nbS, ra.initial, ra.final, ra.transitions)
end = time.time()
print('Finished dpfa & dfa intersection...', end-start)
print(PFA_utils.verifier(at=dpfa, isFile=False))

# Normalize the sub_dpfa to dpfa
start = time.time()
dpfa = PFA_utils.normalizer(sub_dpfa)
end = time.time()
print('Finished sub-dpfa normalization...', end-start)
print(PFA_utils.verifier(at=dpfa, isFile=False))


# Do exact MPS on dpfa
start = time.time()
k_mps = dpfa.MPS()
end = time.time()
print('Finished MPS...', end-start)

total_end = time.time()

print(total_end - total_start, k_mps)
dpfa.print()

"""
print("#########TEST Normalizer########")
initial = np.array([.6, 0])
final = np.array([0.1, 0.2])
transitions = {}
transitions['a'] = np.array([[0.1, 0.0], [0.0, 0.2]])
transitions['b'] = np.array([[0.0,0.3],[0.4,0.0]])

at = PFA.PFA(2, 2, initial, final, transitions)
z=at.initial@np.linalg.inv(np.eye(at.nbS)-(at.transitions['a']+at.transitions['b']))@at.final
print("total value:", z)
print(PFA_utils.verifier(at=at, isFile=False))
at2 = PFA_utils.normalizer(at)
print(PFA_utils.verifier(at=at, isFile=False))
print("total value after normalization:", at2.initial@np.linalg.inv(np.eye(at2.nbS)-(at2.transitions['a']+at2.transitions['b']))@at2.final)
print(at.parse("aaa")/z, at2.parse("aaa"))
at2.print()
"""