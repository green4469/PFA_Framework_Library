from common_header import *

import DFA
import PFA

from DS import Queue, Node



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


        at = PFA.PFA(nbL = nbL, nbS= nbS, initial = initial, final = final, transitions = transitions)
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
        f.close()

def DPFA_generator(nbS, nbL):
    is_DPFA = False

    while not is_DPFA:
        sigma = [str(chr(ord('a')+i)) for i in range(nbL)]

        initial = np.zeros(nbS, dtype=np.float64)
        initial[0] = 1.0

        final = np.zeros(nbS, dtype=np.float64)

        transitions = {}
        for alpha in sigma:
            transitions[alpha] = np.zeros((nbS,nbS), dtype=np.float64)


        for i in range(nbS):
            T = random.randint(0, len(sigma))  # number of outgoint transitions

            # Select T alphabets from sigma
            sigma_T = sigma[:]  # deep copy
            for _ in range(len(sigma) - T):  # Remove nbL - T alphabets from sigma_T
                sigma_T.remove(random.choice(sigma_T))
            assert len(sigma_T) == T, 'wrong num of transitions'

            # The sum of outgoing transitions probabilities + final probability equal to 1
            probs = sum_to_one(T+1)

            final[i] = probs[0]

            for j, alpha in enumerate(sigma_T):
                transitions[alpha][i][random.randint(0, nbS-1)] = probs[j+1]

        at = PFA.PFA(nbL, nbS, initial, final, transitions) 

        if verifier(at=at, isFile=False):
            is_DPFA = True

    return at

"""
def DPFAgenerator(fname, num_state_min = 5, num_state_max = 5):
    alphabets = 'abcdefghijklmnopqrstuvwxyz'
    alphabets = [str(alpha) for alpha in alphabets]
    nbL = len(alphabets)
    nbS = random.randint(num_state_min, num_state_max)
    nbT = 0 # the number of transitions

    initial = [0 for i in range(nbS)]
    idx = random.randint(0,nbS-1)
    initial[idx] = 1

    final = []
    transitions = []

    #alpha_state_comb = list(itertools.product(alphabets, range(nbS)))
    for i in range(nbS):
        T = random.randint(0, len(alphabets))  # the number of outgoing transitions for this state
        nbT += T
        alphabet_list = alphabets[:] # the remain alphabets
        for _ in range(len(alphabets) - T):
            alphabet_list.remove(random.choice(alphabet_list))
        probs = sum_to_one( T + 1 )  # the sum of outgoing transitions probabilities + final probability = 1
        final.append(probs[0])
        for j, alphabet in enumerate(alphabet_list):
            transitions.append((i, alphabet, random.randint(0,nbS-1), probs[j+1]))

    with open(fname, 'w') as f:
        f.write("{} {} {}\n".format(nbS, nbL, nbT))

        for i in range(nbS):
            f.write("{} {}\n".format(initial[i], final[i]))

        for tp in transitions:
            f.write("{} {} {} {}\n".format(tp[0], tp[1], tp[2], tp[3]))
    f.close()
"""


# Verify the generated PFA input files
def verifier(fname=None, at=None, isFile=True):
    if isFile:
        at = parser(fname)
    if at.probability_cond()[0] and at.terminating_cond()[0] and False not in at.get_reachable_state_flag():
        return True
    else:
        if isFile:
            os.remove(fname)
        return False

def pfa2input(pfa, file_name):
    num_states = pfa.nbS
    alphabet_size = len(pfa.alphabets)
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

# Normalize sub-DPFA to DPFA
def from_initial_to_state_string(at, target_state):
    # BFS
    from DS import Node, Queue

    root = Node()
    root.data = (np.argmax(at.initial), '')  # (state, generated_string) pair. For root node, the initial state (unique in (sub-)DPFA), and null string

    q = Queue()
    q.enqueue(root)

    while not q.is_empty():
        current_node = q.dequeue()
        current_state = current_node.data[0]
        current_string = current_node.data[1]

        if current_state == target_state:
            return current_string

        # Find the successive nodes of the current node from the given automaton
        for a, tm in at.transitions.items():
            # Each alphabet has one next state. (Since its sub-DPFA)
            # Find that state
            next_state = np.argmax(tm[current_state])

            new_node = Node()
            new_node.data = (next_state, current_string + a)

            q.enqueue(new_node)

    raise Exception('There exist unreachable state')

def normalizer(at):
    new_initial = np.zeros(at.nbS, dtype=np.float64)
    new_initial[0] = 1.0
    at.initial = new_initial

    new_final = np.zeros(at.nbS, dtype=np.float64)
    new_transitions = {}
    for alpha in at.alphabets:
        new_transitions[alpha] = np.zeros((at.nbS, at.nbS), dtype=np.float64)

    ##
    for current_state in range(at.nbS):
        w = from_initial_to_state_string(at, current_state)
        new_final[current_state] = at.parse(w) / at.prefix_prob2(w)

        for a, tm in at.transitions.items():
            next_state = np.argmax(tm[current_state])

            if tm[current_state][next_state] == 0.0:  # What if there's no next state for this alphabet?
                continue

            new_transitions[a][current_state, next_state] = at.prefix_prob2(w+a) / at.prefix_prob2(w)
    ##

    at.final = new_final
    at.transitions = new_transitions
    return at


def DFA_constructor(w, k, sigma):
    """
    Hamming Automata Construction
    """
    n = len(w)  # the length of input string w

    # Find the number of states from w, k
    nbS = 0
    for i in range(k+1):
        nbS += n+1-i
    nbS += 1  # Consider the sink state

    # Decaler empty initial, transition, final probabilities
    initial = np.zeros(nbS, dtype=np.float64)
    final = np.zeros(nbS, dtype=np.float64)
    transition = {}
    for alphabet in sigma:
        transition[alphabet] = np.zeros((nbS,nbS), dtype=np.float64)
    # Define the final states
    final_index = n
    for i in range(k+1):
        final[final_index] = 1.0
        final_index += n-i

    # Define the initial state
    initial[0] = 1.0

    # Define the transition matrices
    current_state = 0
    w_index = 0

    for i in range(k+1):
        w_index = i
        for j in range(n+1-i):
            for alphabet, tm in transition.items():
                if final[current_state] != 1 and alphabet == w[w_index]:
                    next_state = current_state + 1
                    tm[current_state, next_state] = 1.0
                elif final[current_state] == 1:  # final state goes to sink state
                    next_state = nbS-1
                    tm[current_state, next_state] = 1.0
                else:
                    next_state = current_state + (n + 1 - i)
                    if next_state < nbS:  # check if the state index grows over the limit
                        tm[current_state, next_state] = 1.0
                    else:
                        next_state = nbS-1
                        tm[current_state, next_state] = 1.0
                tm[nbS-1,nbS-1] = 1.0  # sink state
            w_index += 1
            current_state += 1

    return DFA.DFA(nbS, len(sigma), 0, initial, transition, final)


if __name__ == "__main__":
    # Test DFA Constructor
    dfa = DFA_constructor('aaaaaa', 1, ['a', 'b'])
    dfa.print()
    print(dfa.verify_acceptance('aaaaab'))

