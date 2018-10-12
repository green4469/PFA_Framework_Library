import lev
import common_header

import PFA_utils

#subdpfa = lev.LevinshteinAutomata('aba', 2)

subdpfa = PFA_utils.parser('sDPFA.txt')
subdpfa.print()

dpfa = PFA_utils.normalizer(subdpfa)

#dpfa.print()
