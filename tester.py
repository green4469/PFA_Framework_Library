import numpy as np
from numpy.linalg import inv

import math
import copy

from PFA import PFA


# Test Code
def float_catcher(number):
    try:
        return float(number)
    except:
        a, b = map(int, number.split('/'))
        return float(a/b)

def test(input_file, string):
    with open(input_file, 'r') as f:
        print(input_file, string)
        print()
        lines = f.readlines()

        nbL = int(lines[0])
        nbS = int(lines[1])
        initial = np.array([float_catcher(number) for number in lines[2].split(',')], dtype=np.float64)
        final = np.array([float_catcher(number) for number in lines[3].split(',')], dtype=np.float64)
        alphabets = [line.replace('\n', '') for line in lines[4].split(',')]

        transitions = {}
        pos = 5
        for char in alphabets:
            array = []
            for i in range(pos, pos + nbS):
                array.append([float_catcher(number) for number in lines[i].split(',')])
            transitions[char] = np.array(array, dtype=np.float64)
            pos += nbS
        
        at = PFA(nbL, nbS, initial, final, transitions)
        at.print()

        
        print('generate a string:', at.generate())
        print('probability of {}:'.format(string), at.parse(string))
        #print('most probable string:', at.BMPS_exact(0.083))
        print('prefix_prob of {}:'.format(string), at.prefix_prob(string))
        print('prefix_prob2 of {}:'.format(string), at.prefix_prob2(string))
        print('suffix_prob of {}:'.format(string), at.suffix_prob(string))
        print('probability condition:', at.probability_cond())
        print('terminating condition:', at.terminating_cond())
        print('bestpath and bestscore:', at.viterbi(string))
        print('#############################################')
        print()

test('inputs/input.txt','abaabb')
test('inputs/input2.txt', 'a')
test('inputs/input3.txt', 'abaabb')