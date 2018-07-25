import numpy as np

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
    
    def parse(self, string):
        pass

"""
Input Examples from http://pageperso.lif.univ-mrs.fr/~remi.eyraud/scikit-splearn/ 
"""
ex_initial = [1, 0]
ex_final = [0, 1/4]
ex_transitions = [
    [1/2, 1/6, 0, 1/4],  # Ma
    [0, 1/3, 1/4, 1/4],  # Mb
]

ex_automaton = R_Automata(ex_initial, ex_final, ex_transitions)
print('generate a string:', ex_automaton.generate())
print('probability of "abba":',ex_automaton.parse('abba'))