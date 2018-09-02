import numpy as np
class LevenshteinAutomaton:
    def __init__(self, string, n):
        self.string = string
        self.max_edits = n
        self.states = {} #key: levenshtein distance list, value : a state number
        self.counter = 0
        self.transitions = []
        self.accepting_states = []
        
        self.explore(self.start())

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
        """
        Check if the state is accepting state
        """
        return state[-1] <= self.max_edits # [-1] the first index from back

    def can_match(self, state):
        """
        Check if the state is a valid state(the string is acceptable after reaching this state)
        """
        return min(state) <= self.max_edits

    def one_state_transitions(self, state):
        """
        Return OUT transitions(alphabets) of the state
        """
        return set(c for (i,c) in enumerate(self.string) if state[i] <= self.max_edits)
    
    def explore(self, state):
        """
        IN: state(levenshtein distance list of row i;ith character of the string)
        initial call : self.explore(self.start())
        OUT: the next state number
        """
        key = tuple(state) # lists can't be hashed in Python so convert to a tuple
        if key in self.states:
            return self.states[key]
        i = self.counter # current state number
        self.counter += 1
        self.states[key] = i
        if self.is_match(state):
            self.accepting_states.append(i)
        for c in self.one_state_transitions(state) | set(['*']): # {transition} U {*}
            newstate = self.step(state, c)
            if self.can_match(newstate):
                j = self.explore(newstate)
                self.transitions.append((i, c, j))
        return i

    def transitions_redefine(self):
        #These transition matrices may be sparse matrices 
        nbS = self.counter
        old_transitions = self.transitions
        self.transitions = {}
        for c in self.string:
            self.transitions[c] = np.zeros((nbS, nbS), dtype=np.float64)
        self.transitions[''] = np.zeros((nbS,nbS),dtype=np.float64)
        for transition in old_transitions:
            alphabet = transition[1]
            from_state = transition[0]
            to_state = transition[2]
            if alphabet is '*':
                for key in self.transitions:
                    self.transitions[key][from_state,to_state] = 1
            else:
                self.transitions[alphabet][from_state,to_state] = 1
        if np.count_nonzero(self.transitions['']) == 0:
            self.transitions.pop('',None)

        
lev = LevenshteinAutomaton("ab", 1)

"""
state0 = lev.start()
print(state0)
state1 = lev.step(state0, 'w')
print(state1)
state2 = lev.step(state1, 'o')
print(state2)
"""

print("counter = nbS")
print(lev.counter)
print("-------------------------------------")
print("states : value of dict.")
print(lev.states)
print("-------------------------------------")
print("transitions")
print(lev.transitions)
print("-------------------------------------")
print("accepting_states")
print(lev.accepting_states)

lev.transitions_redefine()
print("-------------------------------------")
print("transitions matrices")
print(lev.transitions)
