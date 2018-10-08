"""
input : PFA files
output : files of string that cab be generated by the PFA

pfa file format
input1.txt
input2.txt
...

string file format
string1.txt
string2.txt
...
"""
from PFA import PFA
import PFA_utils
import os

folder = "./inputs/pfa"
for file in os.listdir(folder):
    filepath = os.path.join(folder, file)
    pfa = PFA_utils.parser(filepath) # PFA
    pfa.make_string_file("inputs/string/string{}.txt".format(file[5]),num_of_strings = 100)