import numpy as np
from numpy.linalg import inv

import math
import copy

#import editdistance

from RA import RA

from DS import Queue


class PFA(RA):
    def __init__(self, nbL, nbS, initial, final, transitions):
        super(PFA, self).__init__(nbL, nbS, initial, final, transitions)

        # Precalculate for the reason of complexity.
        M = np.zeros((self.nbS, self.nbS))
        for _, matrix in self.transitions.items():
            M += matrix
        self.Msigma = M

        I = np.eye(self.nbS)
        self.I = I  # Identity Matrix

        self.inv_I_M = inv(I-M)  # Inverse Matrix

        self.u = self.initial @ self.Msigma @ (self.inv_I_M ** 2) @ self.final  # mean of distribution
        self.var = self.initial @ self.Msigma @ (self.I + self.Msigma) @ (self.inv_I_M ** 3) @ self.final \
                    - ( self.initial @ self.Msigma @ (self.inv_I_M ** 2) @ self.final ) ** 2  # variance of distribution

    # Hak-Su
    """
    Viterbi algorithm in "Representing Distributions over Strings with
                          Automata and Grammars", Page 107

    Input           a string w
    Output          the highest probability of w along with one path,
                    the sequence of the states(from initial states to final states) of the path.
    Description     Viterbi algorithm Implementation
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

        for i in range(1,n+1):
            for j in range(self.nbS):
                #below is reducing time complexity but there are some problems with Vpath
                temp_ndarray = V[i-1][:]*self.transitions[string[i-1]].transpose()[j,:]
                V[i][j] = max(temp_ndarray)
                Vpath[i][j] = Vpath[i-1][np.argmax(temp_ndarray)]+str(j)

        #Multiply by the halting probabilities
        bestscore = 0
        bestpath = ""
        temp_ndarray = V[n][:]*self.final[:]
        bestscore = max(temp_ndarray)
        bestpath = Vpath[n][np.argmax(temp_ndarray)]
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

        M_w = np.eye(self.nbS)
        for char in w:
            M_w= M_w @ self.transitions[char]

        return self.initial @ M_w @ self.inv_I_M @ self.final

    def suffix_prob(self, w):
        """
        Input           a PFA, a string w
        Output          the probability of w appearing as a suffix
        Author          Yu-Min Kim
        """

        M_w = np.eye(self.nbS)
        for char in w:
            M_w= M_w @ self.transitions[char]

        return self.initial @ self.inv_I_M @ M_w @ self.final

    def BMPS_exact(self, p):
        """
        Algorithm 2 in sampling-algorithm.pdf

        Input           a PFA, p >= 0
        Output          the string w such that PrA(w) > p or false if there is no such w
        Description     Solve  BMPS(Bounded Most Probable String) which returns the string
                        whose probability is greater than p and length is less than b.
        Complexity      O((b * nbL * (nbs**2)) / p) if all operations are constant

        Author          Yu-Min Kim
        """

        # Calculate bound
        b = math.ceil(self.u + self.var/p)
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

    def MPS(self):
        epsilon = 0.0001
        low = 0.0
        high = 1.0

        # Continuous Bisection Search
        while True:
            mid = (low + high) / 2
            w = self.BMPS_exact(mid)

            # If w is False, then lower the threshold
            if w == False:
                high = mid
            # Else then higher the threshold
            else:
                low = mid

            # If high-low is lower than epsilon and w is not False, then break
            if high - low < epsilon and w != False:
                return w

    def k_MPS(self, x, k):
        """
        Return MPS where the string is within 1 hamming distance with given string x (k = 1)
        Input: an Automaton, a string x, an positive integer k
        Output: MPS under k
        """

        """ Naive algorithm
        strings = []  # all possible strings derived from x within hamming distance 1

        # Make the given string x to a list
        x_list = list(x)

        # Add all possible strings to list
        for i in range(len(x_list)):
            original = x_list[i]
            for char in self.alphabet:
                x_list[i] = char
                strings.append(''.join(x_list))
            x_list[i] = original

        # Delete duplicates and sort
        strings = set(strings)
        strings = list(strings)
        strings.sort()
        print(strings)
        """

        x_list = list(x)  # make input string x to a list

        # Prefix probabilities
        # O(Sigma)
        prefix_list = []
        initial = self.initial
        prefix_list.append(initial)

        for char in x_list: 
            initial = initial @ self.transitions[char]
            prefix_list.append(initial)

        # Infix probabilities
        # O(n^3)
        infix_dict = {}
        n = len(x)
        for i in range(n): 
            for j in range(i+1, n):
                infix_prob = self.I  # Start with identity matrix
                for idx in range(i+1, j):
                    infix_prob = infix_prob @ self.transitions[x_list[idx]]
                infix_dict[(i,j)] = infix_prob  # Later, reference like infix_list[(i, j)]
                

        # Suffix probabilities  
        # O(Sigma)
        suffix_list = []
        final = self.final
        suffix_list.append(final)
        x_list.reverse()  # Start from the end
        for char in x_list:  
            final = self.transitions[char] @ final
            suffix_list.append(final)
        suffix_list.pop()
        suffix_list.reverse()
        x_list.reverse()  # Make the list original order


        # When k = 1
        if k == 1:
            x_list = list(x)

            # Calculate all probabilities of possible strings where (k=1, x)
            most_prob = 0
            MPS = []
            for i in range(len(x_list)):
                for char in self.alphabet:
                    prob = prefix_list[i] @ self.transitions[char] @ suffix_list[i]
                    if prob > most_prob:
                        most_prob = prob
                        MPS = x_list[:]
                        MPS[i] = char

            return ''.join(MPS)


        # Find all possible combinations when k using cartesian product
        # nCk, Sigma^k?
        import itertools
        alpha_comb = list(itertools.product(self.alphabet, repeat=k))  # Cartesian product for repeat k, e.g., A = ['a', 'b']; when k = 3; A x A x A
        pos_comb = itertools.combinations(range(n), k)  # Combinations for posstible k positions

        # Calculate probabilities
        # O(nCk * Sigma^k * k)
        most_prob = 0
        MPS = []
        for pos_tuple in pos_comb:
            for alpha_tuple in alpha_comb:
                MPS_candidate = x_list[:]

                ###
                prob = prefix_list[pos_tuple[0]]
                for i in range(k-1):
                    prob = prob @ self.transitions[alpha_tuple[i]]
                    MPS_candidate[pos_tuple[i]] = alpha_tuple[i]
                    prob = prob @ infix_dict[(pos_tuple[i], pos_tuple[i+1])]
                prob = prob @ self.transitions[alpha_tuple[k-1]]
                MPS_candidate[pos_tuple[k-1]] = alpha_tuple[k-1]
                prob = prob @ suffix_list[pos_tuple[k-1]]
                ###

                if prob > most_prob:
                    most_prob = prob
                    MPS = MPS_candidate[:]

        # Find k-MPS
        MPS = ''.join(MPS)

        return MPS

        
