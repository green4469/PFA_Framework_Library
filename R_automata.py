import numpy as np

alphabet_index = {'a' : 0, 'b' : 1, 'c' : 2, 'd' : 3, 'e' : 4, 'f' : 5, 'g': 6, 'h' : 7, 'i': 8, 'j': 9, 'k': 10,
                  'l': 11, 'm': 12, 'n':13, 'o':14, 'p':15, 'q':16, 'r':17, 's':18, 't':19, 'u':20,
                  'v':21, 'w':22, 'x':23, 'y':24, 'z':25} 

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

    def generate(self):
        pass

    """
    Forward algorithm in "Representing Distributions over Strings with
                          Automata and Grammars", Page 105
    """
    def parse(self, string):
        #Algorithm 5.2: FORWARD.
        n = int(len(string))
        Q = self.nbS
        F = np.zeros((n+1, Q+1), dtype=np.float64)
        
        #initialize
        for j in range(1,Q+1):
            F[0][j] = self.initial[j-1]
            for i in range(1,n+1):
                F[i][j] = 0

        for i in range(1,n+1):
            for j in range(1,Q+1):
                for k in range(1,Q+1):
                    number = string[i-1]
                    index = alphabet_index[number]
                    F[i][j] += F[i-1][k]*self.transitions[index][k-1][j-1]
        #Algorithm 5.3: Computing the probability of a string with FORWARD.
        T = 0
        for j in range(1,Q+1):
            T += F[n][j]*self.final[j-1]
        return T
            
            

"""
Input Examples from http://pageperso.lif.univ-mrs.fr/~remi.eyraud/scikit-splearn/ 
"""
ex_initial = [1, 0]
ex_final = [0, 1/4]
ex_transitions = [
    [[1/2, 1/6],
     [0,   1/4] ],  # Ma
    [[0,   1/3],
     [1/4, 1/4] ]  # Mb
]

ex_automaton = R_Automata(2,2,ex_initial, ex_final, ex_transitions)
print('generate a string:', ex_automaton.generate())
print('probability of "aba":',ex_automaton.parse('aba'))
