"""
instructions
generate_input.py dpfa num_of_files
generate_input.py string_for_pfa
generate_input.py string num_of_strings

pfa file format
pfa/input1.txt
pfa/input2.txt
...

string file format
string/string1.txt
string/string2.txt
...

strings file format
strings.txt
"""
from PFA import PFA
import PFA_utils
import os
import sys
import random

def DPFAgenerate(n):
    idx = 0
    while(True):
        fname = 'inputs/pfa/input{}.txt'.format(idx)
        PFA_utils.DPFAgenerator(fname)
        print("Created an input file")
        if PFA_utils.verifier(fname):
            print("Verified #{} input.".format(idx))
            idx += 1
            if idx == n:
                return
        else:
            print("Failed to verify")

def dpfa():
    DPFAgenerate(int(sys.argv[2]))

def string_for_PFA():
    folder = "./inputs/pfa"
    for file in os.listdir(folder):
        filepath = os.path.join(folder, file)
        pfa = PFA_utils.parser(filepath) # PFA
        pfa.make_string_file("inputs/string/string{}.txt".format(file[5]),num_of_strings = 10)

def string(num_of_alphabet):
    f = open("inputs/strings.txt", "w")
    alphabets = 'abcdefghijklmnopqrstuvwxyz'
    alphabets = alphabets[0:num_of_alphabet]    
    alphabets = [str(alpha) for alpha in alphabets]
    for i in range(1,11):
        for _ in range(100):
            length = i*5
            one_string = ""
            for _ in range(length):
                one_string += random.choice(alphabets)
            f.write("{}\n".format(one_string))
    f.close()   

if __name__ == "__main__":
    # generate_input.py dpfa num_of_files
    if sys.argv[1] == "dpfa":
        dpfa()
    # generate_input.py string_for_pfa
    elif sys.argv[1] == "string_for_pfa":
        string_for_PFA()

    elif sys.argv[1] == "string":
        string(int(sys.argv[2]))
    