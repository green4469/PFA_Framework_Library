""" This module defines Levinshtein Automata class """

from common_header import *
import DFA 
import PFA
import PFA_utils

class LevenshteinAutomaton(DFA.DFA):
    def __init__(self, string, max_edits):
        super(LevenshteinAutomaton, self).__init__(nbL = 0, nbS = 0, initial_state = 0, states = [],
                                                   transitions = {}, final_states = [])
        self.string = string
        self.max_edits = max_edits
        self.states = {} #key: levenshtein distance list, value : a state number
        """
        alphabets = []
        for alphabet in string:
            alphabets.append(alphabet)
        self.alphabets = list(set(alphabets))
        """
        self.alphabets = [str(alpha) for alpha in alphabets]
        self.explore(self.start())

    def start(self):
        return list(range(len(self.string)+1))

    def step(self, state, c):
        """
        M[i,j] := levenshtein distance between A[0..i] and B[0..j], A : input string , B : generated string(initial : empty string) + c
        input : current(row i:ith character of string) levenshtein distance list(state), alphabet c
        output : next(row i+1) levenshtein distance list(new_state)
        """
        new_state = [state[0]+1]
        for i in range(len(state)-1):
            cost = 0 if self.string[i] == c else 1
            new_state.append(min(new_state[i]+1, state[i]+cost, state[i+1]+1))
            # new_state[i]+1 : B can be A by inserting character c
            # state[i] + cost : the distance is increase if the last characters of two string A and B are different
            #                   ( B can be A modify c to the last character of A )
            # state[i+1]+1 : B can be A by deleting c
        return [min(x,self.max_edits+1) for x in new_state]

    def is_match(self, state):
        """
        Check if the state is accepting state
        """
        # if the distance of last character is less than k, the state can be accepting state
        return state[-1] <= self.max_edits # [-1] the first index from back

    def can_match(self, state):
        """
        Check if the state is a valid state(the string is acceptable after reaching this state)
        """
        # if the minimun distance of the distances less than or equal to k, the string is acceptable after reaching this state
        # otherwise( min(state) > k) there is no possibility to accept this string.
        return min(state) <= self.max_edits 
    """
    ### not used ###
    def one_state_transitions(self, state):
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
        i = self.nbS # current state number
        self.nbS += 1
        self.states[key] = i
        if self.is_match(state):
            self.final_states.append(i)
        #for c in sorted(self.one_state_transitions(state) + ['*']): # {transition} U {*}
        for c in list(set(self.string)) + ['*']: # {transition} U {*}
            newstate = self.step(state, c)
            if self.can_match(newstate):
                j = self.explore(newstate)
                self.transitions[(i,c)] = j
        return i

    def transitions_matrix(self):
        #These transition matrices may be sparse matrices
        nbS = self.nbS
        old_transitions = self.transitions
        self.transitions = {}
        for alphabet in alphabets:
            self.transitions[alphabet] = np.zeros((nbS, nbS), dtype=np.float64)
        self.transitions[''] = np.zeros((nbS,nbS),dtype=np.float64)
        for (current_state, alphabet) in old_transitions:
            next_state = old_transitions[(current_state, alphabet)]
            """
            alphabet = transition[1]
            from_state = transition[0]
            to_state = transition[2]
            """
            if alphabet is '*':
                for key in self.transitions:
                    self.transitions[key][current_state,next_state] = 1
            else:
                self.transitions[alphabet][current_state,next_state] = 1
        if np.count_nonzero(self.transitions['']) == 0:
            self.transitions.pop('',None)
        else: #remove '' transitions
            #idx_pair_list : index pair (x,y) which is transitions[''] == 1
            idx_pair_list = [(ix,iy) for ix, row in enumerate(self.transitions['']) for iy, i in enumerate(row) if i == 1.0]
            keys = set(self.alphabets)
            excludes = set([''])
            for pair in idx_pair_list:
                for key in keys.difference(excludes):
                    self.transitions[key][pair[0],pair[1]] = 1.0
            self.transitions.pop('',None)

if __name__ == "__main__":
    """ Levinshtein Automata class unit-test code """
    at = PFA_utils.parser("./inputs/pfa/input0.txt")
    print('parsing done')
    start_time = time.time()
    lev = LevenshteinAutomaton("abaaab", 3)
    print('levenshtein done')
    r = at.intersect_with_DFA(lev)
    end_time = time.time()
    print('intersect done')
    """
    state0 = lev.start()
    print(state0)
    state1 = lev.step(state0, 'w')
    print(state1)
    state2 = lev.step(state1, 'o')
    print(state2)
    """

    np.set_printoptions(precision=2)
    print("nbS")
    print(lev.nbS)
    print("-------------------------------------")
    print("states : value of dict.")
    print(lev.states)
    print("-------------------------------------")
    print("transitions")
    #print(lev.transitions)
    print("-------------------------------------")
    print("accepting_states")
    #print(lev.final_states)

    lev.transitions_matrix()
    print("-------------------------------------")
    print("transitions matrices")
    #print(lev.transitions)
    print("-------------------------------------")

    
    print("nbS")
    print(r.nbS)
    print("-------------------------------------")
    print("transitions")
    #print(r.transitions)
    print("-------------------------------------")
    print("intial_states")
    #print(r.initial)
    print("-------------------------------------")
    print("final_states")
    #print(r.final)
    print("-------------------------------------")
    print("total running time of lev + intersection")
    print(end_time - start_time)
