class LevenshteinAutomaton:
    def __init__(self, string, n):
        self.string = string
        self.max_edits = n

    def start(self):
        return range(len(self.string)+1)

    def step(self, state, c):
        """
        M[i,j] := levenshtein distance between A[0..i] and B[0..j]
        input : current(row i:ith character of string) levenshtein distance list(state), alphabet c
        output : next(row i+1) levenshtein distance list(new_state)
        """
        new_state = [state[0]+1]
        for i in range(len(state)-1):
            cost = 0 if self.string[i] == c else 1
            new_state.append(min(new_state[i]+1, state[i]+cost, state[i+1]+1))
        return [min(x,self.max_edits+1) for x in new_state]

    def is_match(self, state):
        return state[-1] <= self.max_edits # [-1] the first index from back

    def can_match(self, state):
        return min(state) <= self.max_edits

    def transitions(self, state):
        return set(c for (i,c) in enumerate(self.string) if state[i] <= self.max_edits)


counter = [0] # list is a hack for mutable lexical scoping
states = {} # key: levenshtein distance list, value : a state number
transitions = [] 
matching = []

lev = LevenshteinAutomaton("woof", 1)

def explore(state):
    """
    IN: state(levenshtein distance list of row i;ith character of the string)
    OUT: the next state number
    """
    key = tuple(state) # lists can't be hashed in Python so convert to a tuple
    if key in states:
        return states[key]
    i = counter[0] # current state number
    counter[0] += 1
    states[key] = i
    if lev.is_match(state):
        matching.append(i)
    for c in lev.transitions(state) | set(['']):
        newstate = lev.step(state, c)
        j = explore(newstate)
        transitions.append((i, c, j))
    return i

explore(lev.start())
"""
state0 = lev.start()
print(state0)
state1 = lev.step(state0, 'w')
print(state1)
state2 = lev.step(state1, 'o')
print(state2)
"""

print("counter")
print(counter)
print("-------------------------------------")
print("states")
print(states)
print("-------------------------------------")
print("transition")
print(transitions)
print("-------------------------------------")
print("matching")
print(matching)
