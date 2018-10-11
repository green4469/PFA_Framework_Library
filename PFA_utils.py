from common_header import *

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


        at = PFA(nbL = nbL, nbS= nbS, initial = initial, final = final, transitions = transitions)
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


# Verify the generated PFA input files
def verifier(fname):
    at = parser(fname)
    if at.probability_cond()[0] and at.terminating_cond()[0] and False not in at.get_reachable_state_flag():
        return True
    else:
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
    new_final = np.zeros(at.nbS, dtype=np.float64)
    new_transitions = {}
    for alpha in at.alphabets:
        new_transitions[alpha] = np.zeros((at.nbS, at.nbS), dtype=np.float64) 

    for current_state in range(at.nbS):
        w = from_initial_to_state_string(at, current_state)
        print("#######")
        print(w)
        print(at.parse(w))
        print(at.prefix_prob(w))
        print("#######")
        new_final[current_state] = at.parse(w) / at.prefix_prob(w)

        for a, tm in at.transitions.items():
            next_state = np.argmax(tm[current_state])
            new_transitions[a][current_state, next_state] = at.prefix_prob(w+a) / at.prefix_prob(w)

    at.final = new_final
    at.transitions = new_transitions
    return at

