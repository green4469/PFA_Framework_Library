import numpy as np
from numpy.linalg import inv

import math
import copy

from RA import RA

from DS import Queue


class PFA(RA):
    def __init__(self, nbL=0, nbS=0, initial=[], final=[], transitions=[]):
        super(PFA, self).__init__(nbL, nbS, initial, final, transitions)

    # Hak-Su
    """
    Viterbi algorithm in "Representing Distributions over Strings with
                          Automata and Grammars", Page 107
    """
    def viterbi(self, string):
        #Algorithm 5.6: VITERBI
        n = len(string)
        V = np.zeros((n+1,self.nbS),np.float64)
        Vpath = [["" for i in range(self.nbS)] for i in range(n+1)]
        #initialise
        V[0,:] = self.initial.transpose()[:]
        for i in range(self.nbS):
            Vpath[0][i] = str(i)
        #for j in range(self.nbS):
        #    V[0,j] = self.initial[j]

        for i in range(1,n+1):
            for j in range(self.nbS):
                #below is reducing time complexity but there are some problems with Vpath
                temp_ndarray = V[i-1][:]*self.transitions[string[i-1]].transpose()[j,:]
                V[i][j] = max(temp_ndarray)
                Vpath[i][j] = Vpath[i-1][np.argmax(temp_ndarray)]+str(j)
                """
                #below is the original algorithm
                for k in range(self.nbS):
                   if V[i][j] < V[i-1][k]*self.transitions[string[i-1]][k,j]:
                        V[i][j] = V[i-1][k]*self.transitions[string[i-1]][k,j]
                        Vpath[i][j] = Vpath[i-1][k]+str(j) 
                """
        #Multiply by the halting probabilities
        bestscore = 0
        bestpath = ""
        temp_ndarray = V[n][:]*self.final[:]
        bestscore = max(temp_ndarray)
        bestpath = Vpath[n][np.argmax(temp_ndarray)]
        """
        for j in range(self.nbS):
            if V[n][j]*self.final[j] > bestscore:
                bestscore = V[n][j]*self.final[j]
                bestpath = Vpath[n][j]
        """
        return bestpath, bestscore #Is the bestsocre necessary to be returned?

    # Myeong-Jang
    def probability_cond(self):
        if abs(np.sum(self.initial) - 1.0) > 1e-8:
            return False, "Wrong initial prob"
        for q in range(self.nbS):
            total_prob = .0
            for transition in self.transitions.values():
                total_prob += transition[q,:].sum()
            total_prob += self.final[q]
            if abs(total_prob - 1.0) > 1e-8:
                return False, "Wrong transition prob at state %d"%(q)
        return True, ""

    def get_reachable_state_flag(self):
        # obtain all reachable states
        reachable_flag = self.initial.astype(np.bool)
        while True:
            prev_flag = copy.deepcopy(reachable_flag)
            reachable_states = np.nonzero(reachable_flag)
            for transition in self.transitions.values():
                reachable_flag += np.sum(transition[reachable_states].astype(np.bool), axis=0).astype(np.bool)
            if not np.sum(np.logical_xor(prev_flag, reachable_flag)):
                return reachable_flag
            else:
                del prev_flag
                continue


    def terminating_cond(self):
        """
        For all reachable states, there exists a path to a state which can be final
        """
        unreachable_flag = np.ones((self.nbS,), dtype=np.bool) ^ self.get_reachable_state_flag()
        terminating_flag = self.final.astype(np.bool) + unreachable_flag
        while True:
            prev_flag = copy.deepcopy(terminating_flag)
            terminating_states = np.nonzero(terminating_flag)
            for transition in self.transitions.values():
                terminating_flag += np.sum(transition[:,terminating_states[0]].astype(np.bool), axis=1).astype(np.bool)
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


    # Yu-Min
    def prefix_prob(self, w):
        """
        Input           a PFA, a string w
        Output          the probability of w appearing as a prefix
        Author          Yu-Min Kim
        Description     Simple Implementation
        """

        result = self.initial
        
        for char in w:
            result = result @ self.transitions[char]

        return result.sum()

    def prefix_prob2(self, w):
        """
        Input           a PFA, a string w
        Output          the probability of w appearing as a prefix
        Author          Yu-Min Kim
        Description     More Complex Implementation
        """

        M = np.zeros((self.nbS, self.nbS))
        for _, matrix in self.transitions.items():
            M += matrix

        I = np.eye(self.nbS)

        M_w = np.eye(self.nbS)
        for char in w:
            M_w= M_w @ self.transitions[char]

        return self.initial @ M_w @ inv(I - M) @ self.final

    def suffix_prob(self, w):
        """
        Input           a PFA, a string w
        Output          the probability of w appearing as a suffix
        Author          Yu-Min Kim
        """

        M = np.zeros((self.nbS, self.nbS))
        for _, matrix in self.transitions.items():
            M += matrix

        I = np.eye(self.nbS)

        M_w = np.eye(self.nbS)
        for char in w:
            M_w= M_w @ self.transitions[char]

        return self.initial @ inv(I - M) @ M_w @ self.final

    def BMPS_exact(self, p):
        """
        Algorithm 2 in sampling-algorithm.pdf
        
        Input           a PFA, p >= 0
        Output          the string w such that PrA(w) > p or false if there is no such w
        Description     Solve the decision problem BMPS(Bounded Most Probable String)
                        which returns the string whose probability is greater than p and
                        length is less than b.
        Complexity      O((b * nbL * (nbs**2)) / p) if all operations are constant

        Author          Yu-Min Kim
        """

        def calculate_b(p):
            """
            b is the bound parameter which constraints the length of the string w.
            This function calculates the value of b.
            """
            M = np.zeros((self.nbS, self.nbS))
            for _, matrix in self.transitions.items():
                M += matrix
            
            I = np.eye(self.nbS)

            u = self.initial @ M @ (inv(I - M) ** 2) @ self.final

            var = self.initial @ M @ (I + M) @ (inv(I - M) ** 3) @ self.final \
                        - ( self.initial @ M @ (inv(I - M) ** 2) @ self.final ) ** 2 

            return  math.ceil(u + var / p)

        # Calculate bound
        b = calculate_b(p)
        #print('b value', b)

        # Initially, the result string is empty string (lambda)
        w = ''

        # Instantiate a Queue
        Q = Queue()

        # The probability of lambda string
        p_0 = self.initial @ self.final
        
        # If the probability of lambda stirng is larger than p, then return it.
        if p_0 > p:
            return w
        
        # Enqueue the probability of the lambda string
        Q.enqueue((w, self.initial))

        while not Q.is_empty():
            # w is string and V is a matrix
            w, V = Q.dequeue().data
            #print('current string', w)

            for char in self.alphabet:
                #print('alphabet', char)
                V_new = V @ self.transitions[char]

                if V_new @ self.final > p:
                    return w + char

                if len(w) < b and V_new.sum() > p:
                    #print('Enqueue!', w + char)
                    Q.enqueue((w + char, V_new))
        
        return False
