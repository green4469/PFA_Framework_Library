import numpy as np
import math
import copy

class R_Automata(object):

    def __init__(self, nbL=0, nbS=0, initial=[], final=[], transitions=[]):
        """
        Constructor of R automaton
        """
        self.nbL = nbL
        self.nbS = nbS
        self.initial = initial
        self.final = final
        self.transitions = transitions

    """
    Forward algorithm in "Representing Distributions over Strings with
                          Automata and Grammars", Page 105
    """
    def parse(self, string):
        #Algorithm 5.2: FORWARD.
        n = int(len(string))
        Q = self.nbS
        F = np.zeros((n+1, Q), dtype=np.float64) #F[n][s] = Pr_A(x,q_s), string x = a_1 a_2 ... a_n

        #initialize
        F[0] = self.initial

        for i in range(1,n+1):
            key = string[i-1]
            F[i] += F[i-1]@self.transitions[key][:,:]

        #Algorithm 5.3: Computing the probability of a string with FORWARD.
        T = F[n]@np.transpose(self.final)

        return T


# a@b

class PFA(R_Automata):
    def __init__(self, nbL=0, nbS=0, initial=[], final=[], transitions=[]):
        super(PFA, self).__init__(nbL, nbS, initial, final, transitions)

    def probability_cond(self):
        if abs(np.sum(self.initial), 1.0) > 1e-8:
            return False, "Wrong initial prob"
        for q in range(self.nbS):
            total_prob = .0
            for transition in self.transitions.values():
                total_prob += transition[q,:].sum()
            total_prob += self.final[q]
            if abs(total_prob, 1.0) > 1e-8:
                return False, "Wrong transition prob at state %d"%(q)
        return True, ""

    def get_reachable_state_indices(self):
        # obtain all reachable states
        reachable_flag = self.initial.astype(np.bool)
        while True:
            prev_flag = copy.deepcopy(reachable_flag)
            reachable_states = np.nonzero(reachable_flag)
            for transition in self.transitions.values():
                reachable_flag += np.sum(transition[reachable_states,:].astype(np.bool), axis=1)
            if not np.sum(np.logical_xor(prev_flag, reachable_flag)):
                return reachable_states
            else:
                del prev_flag
                continue


    def terminating_cond(self):
        """
        For all reachable states, there exists a path to a state which can be final
        """
        unreachable_flag = np.ones((self.nbS,), dtype=np.bool) ^ self.get_reachable_state_indices()
        terminating_flag = self.final.astype(np.bool) + unreachable_flag
        while True:
            prev_flag = copy.deepcopy(terminating_flag)
            terminating_states = np.nonzero(terminating_flag)
            for transition in self.transitions.values():
                terminating_flag += np.sum(transition[:,terminating_states].astype(np.bool), axis=0)
            if not np.sum(np.logical_xor(prev_flag, terminating_flag)):
                if False in terminating_flag:
                    return False, "nonterminating states %s"%(str([i for i in range(self.nbS) if i not in terminating_states]))
                else:
                    return True, ""
            else:
                del prev_flag
                continue



    def generate(self):
        s = np.random.choice(self.nbS, p=self.initial)
        generated = ""
        while True:
            # P(x) = sum_E P(x|E)p(E)
            # get an alphabet
            n_a = np.random.choice(self.nbL + 1,
                                 p=[pr_a[s].sum() for pr_a in self.transitions.values()] + [self.final[s]])
            if n_a == self.nbL:
                return generated
            else:
                a = list(self.transitions.keys())[n_a]
                generated += a

            # get the next state
            s = np.random.choice(self.nbS,
                                 p=self.transitions[a][s, :] / np.sum(self.transitions[a][s,:]))
        return generated




"""
Input Examples from http://pageperso.lif.univ-mrs.fr/~remi.eyraud/scikit-splearn/
"""
ex_initial = np.array([1, 0], dtype=np.float64)
ex_final = np.array([0, 1/4], dtype=np.float64)

ex_transitions = {
    'a': np.array([[1/2, 1/6],
                    [0,  1/4]], dtype=np.float64),
    'b': np.array([[0,   1/3],
                    [1/4, 1/4]], dtype=np.float64)
}

ex_automaton = PFA(2,2,ex_initial, ex_final, ex_transitions)
print('generate a string:', ex_automaton.generate())
print('probability of "aba":',ex_automaton.parse('aba'))
