import numpy as np
import math

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
        if abs(np.sum(initial), 1.0) > 1e-8:
            return False, "Wrong initial prob"
        for q in range(self.nbS):
            total_prob = .0
            for _, transition in self.transitions.items():
                total_prob += transition[q,:].sum()
            total_prob += self.final[q]
            if abs(total_prob, 1.0) > 1e-8:
                return False, "Wrong transition prob at state %d"%(q)
        return True, ""

    def halting_cond(self):
        pass

    def generate(self):
        s = np.random.choice(self.nbS, p=self.initial)
        generated = ""
        while True:
            # P(x) = sum_E P(x|E)p(E)
            # get an alphabet
            a = np.random.choice(self.nbL + 1,
                                 p=[pr_a[s].sum() for pr_a in self.transitions.values()] + [self.final[s]])
            if a == self.nbL:
                return generated
            else:
                generated += a

            # get the next state
            s = np.random.choice(self.nbS,
                                 p=self.transitions[a][s])
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

ex_automaton = R_Automata(2,2,ex_initial, ex_final, ex_transitions)
print('generate a string:', ex_automaton.generate())
print('probability of "aba":',ex_automaton.parse('aba'))
