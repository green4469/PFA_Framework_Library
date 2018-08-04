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

    """
    Forward algorithm in "Representing Distributions over Strings with
                          Automata and Grammars", Page 105
    """
    def parse(self, string):
        if 'epsilon' in self.transitions:
            return "ERROR(There are epsilon transitions in the automata. Use the function 'epsilon_transition_removal' before parsing)"
        #Algorithm 5.2: FORWARD.
        n = int(len(string))
        Q = self.nbS
        F = np.zeros((n+1, Q), dtype=np.float64) #F[n][s] = Pr_A(x,q_s), string x = a_1 a_2 ... a_n
        
        #initialize
        F[0,:] = self.initial[:]

        for i in range(1,n+1):
            key = string[i-1]
            F[i,:] += F[i-1,:]@self.transitions[key][:,:]

        #Algorithm 5.3: Computing the probability of a string with FORWARD.
        T = F[n,:]@np.transpose(self.final[:])
        return T

    def epsilon_transition_removal(self):
        """
        Step 1: If there is more than one initial state, add a new initial state and epsilon-transitions
                from this state to each of the previous initial states,
                with probability equal to that of the state being initial.
        """
        #Algorithm 5.7: Transforming the epsilon-PFA into a epsilon-PFA with just one initial state.
        if np.count_nonzero(self.initial) > 1:
            Q = self.nbS
            Q_prime = Q + 1
            initial_original = self.initial[:]

            #new initial probability with Ip(q_0) = 1
            initial_prime = np.zeros((Q_prime), np.float64)
            initial_prime[0] = 1

            #new final probability Fp(q_0) = 0
            final_prime = np.zeros((Q_prime), np.float64)
            final_prime[1:Q_prime] = self.final[:]

            #return new Automata
            self.nbS = Q_prime
            self.initial[:] = initial_prime[:]
            self.final[:] = final_prime[:]
            
            #new transition with just one initial state q_0
            for key in self.transitions.keys():
                temp = self.transitions[key][:,:]
                self.transitions[key] = np.zeros((Q_prime,Q_prime), np.float64)
                self.transitions[key][1:Q_prime, 1:Q_prime] = temp[:,:]
            self.transitions['epsilon'][0,1:Q_prime] = initial_original[:]
        """
        Step 2: Algorithm 5.8 iteratively removes a epsilon-loop if there is one,
                and if not the epsilone-transition with maximal extremity.
        """
        #Algorithm 5.8: Eliminating epsilon-transitions
        Q = self.nbS
        #while 'there still are epsilon-transitions' do
        if 'epsilon' in self.transitions:
            while np.count_nonzero(self.transitions['epsilon']) > 0:
                #if there exists a epsilon-loop (q,epsilon,q, P) then
                for i in range(self.transitions['epsilon'].shape[0]):
                    if self.transitions['epsilon'][i][i] > 0:
                        #for all transitions(q,a,q') , (a,q') != (epsilon,q) do
                        for key in self.transitions.keys():
                            for j in range(Q):
                                if key != 'epsilon' or j != i: # (a,q') != (epsilon, q)
                                    self.transitions[key][i][j] *= (1/(1-self.transitions['epsilon'][i][i]))
                        self.final[i] *= 1/(1-self.transitions['epsilon'][i][i])
                        self.transitions['epsilon'][i][i] = 0
                # there are no epsilon-loops
                # let (q,epsilon,q_m) b a epsilon-transition with m maximal
                m = 0
                for i in range(self.transitions['epsilon'].shape[0]):
                    if self.transitions['epsilon'].transpose()[i].any() > 0:
                        m = i
                for n in range(m):
                    self.transitions['epsilon'][:,n] += self.transitions['epsilon'][:,m] * self.transitions['epsilon'][m][n]
                for key in self.transitions.keys():
                    if key != 'epsilon':
                        for n in range(Q):
                            self.transitions[key][:,n] += self.transitions['epsilon'][:,m]*self.transitions[key][m][n]
                self.final += self.transitions['epsilon'][:,m]*self.final[m]
                self.transitions['epsilon'][:,m] = 0
            self.transitions.pop('epsilon',None)
        ####### 현재 객체 안의 값 바꾸지 말고 return 으로 객체 반환하기(현재 객체는 유지)??

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
                for k in range(self.nbS):
                    if V[i][j] < V[i-1][k]*self.transitions[string[i-1]][k,j]:
                        V[i][j] = V[i-1][k]*self.transitions[string[i-1]][k,j]
                        Vpath[i][j] = Vpath[i-1][k]+str(j)
        #Multiply by the halting probabilities
        bestscore = 0
        bestpath = ""
        for j in range(self.nbS):
            if V[n][j]*self.final[j] > bestscore:
                bestscore = V[n][j]*self.final[j]
                bestpath = Vpath[n][j]
        return bestpath        
    
# a@b

if __name__ == "__main__":
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
    """
    #example in Chapter 5.2 Probabilistic automata P.113
    ex_initial = np.array([1, 0, 0, 0], dtype=np.float64)
    ex_final = np.array([0, 0.4, 0.2, 0.6], dtype=np.float64)
    ex_transitions = {
        'a': np.array([ [0, 0, 0.5, 0],
                        [0, 0, 0, 0.1],
                        [0, 0, 0, 0.1],
                        [0, 0, 0, 0.2]], dtype=np.float64),
        'epsilon' : np.array([[0, 0.5, 0, 0],
                              [0, 0.5, 0, 0],
                              [0, 0.2, 0.5, 0],
                              [0, 0, 0.2, 0]], dtype=np.float64)
    }

    
    ex_automaton = R_Automata(2,4,ex_initial, ex_final, ex_transitions)
    print('probability of "',string,'":',ex_automaton.parse(string))
    ex_automaton.epsilon_transition_removal()
    """
    """
    string = 'a'
    ex_automaton = R_Automata(2,2,ex_initial, ex_final, ex_transitions)
    print('generate a string:', ex_automaton.generate())
    #ex_automaton.epsilon_transition_removal()
    print('probability of "',string,'":',ex_automaton.parse(string))
    """

    #example of Fig.5.4 in Chapter 5.2 Probabilistic automata P.108
    ex_initial = np.array([0.4, 0.6, 0, 0], dtype = np.float64)
    ex_final = np.array([0, 0.6, 0.5, 0.3], dtype = np.float64)
    ex_transitions = {
        'a' : np.array([ [0, 0.5, 0, 0],
                         [0, 0, 0, 0.2],
                         [0, 0, 0.3, 0],
                         [0, 0, 0, 0] ], dtype = np.float64),
        'b' : np.array([[0.5, 0, 0, 0],
                        [0, 0, 0.2, 0],
                        [0, 0, 0, 0.2],
                        [0.5, 0.2, 0, 0]], dtype = np.float64)
        }
    ex_automaton = R_Automata(2, 4, ex_initial, ex_final, ex_transitions)
    print("bestpath(the sequence of the number of the states):",ex_automaton.viterbi('ab'))
