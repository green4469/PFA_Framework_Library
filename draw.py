from graphviz import Digraph
from common_header import *
import sys
from PyQt5 import uic, QtGui
from PyQt5.QtWidgets import *


form_class = uic.loadUiType("draw.ui")[0]
class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.k=0
        self.lineEdit.returnPressed.connect(self.enter_pressed)
        self.lineEdit_2.returnPressed.connect(self.enter_pressed)
        self.lineEdit_2.setText("0")
        self.lineEdit_2.textChanged.connect(self.hamming_distance)
        self.draw.setStyleSheet("background-color:#ffffff;")
        self.tableWidget.cellClicked.connect(self.clicked)
        self.tableWidget.setColumnWidth(1,900)
        self.dpfa = None

    def enter_pressed(self):
        start_time = time.time()
        # make hamming automaton
        hamming = PFA_utils.DFA_constructor(str(self.lineEdit.text()), self.k, list(set(self.lineEdit.text())))
        
        # intersect hamming & pfa -> sub_pfa
        input_dpfa = PFA_utils.DPFA_generator(2, 2)
        ra = input_dpfa.intersect_with_DFA(hamming)
        sub_dpfa = PFA.PFA(ra.nbL, ra.nbS, ra.initial, ra.final, ra.transitions)

        # normalize sub_pfa -> dpfa
        self.dpfa = PFA_utils.normalizer(sub_dpfa)

        # Do exact MPS on dpfa
        k_mps = self.dpfa.MPS()
        end_time = time.time()
        load_time = end_time - start_time
        # initialize the table
        self.tableWidget.clearContents()
        self.tableWidget.setRowCount(0)

        self.tableWidget.insertRow( self.tableWidget.rowCount() )
        self.tableWidget.setItem( self.tableWidget.rowCount()-1, 0, QTableWidgetItem(str(round(self.dpfa.parse(k_mps),10))))
        self.tableWidget.setItem( self.tableWidget.rowCount()-1, 1, QTableWidgetItem(k_mps))
        
        # update drawing
        makePNG(self.dpfa)
        pixmap = QtGui.QPixmap("Digraph.gv.png") 
        self.draw.setPixmap(pixmap)
        self.textBrowser.setText(str(load_time)+' seconds')
        
    def hamming_distance(self):
        self.k = int(self.lineEdit_2.text())

    def clicked(self, row, col):
        item = self.tableWidget.item(row, 1).text()
        emphasizePNG(self.dpfa, item)
        pixmap = QtGui.QPixmap("Digraph.gv.png") 
        self.draw.setPixmap(pixmap)

def makePNG(RA):
    dot = Digraph(comment='PFA', format='png')
    for i in range(RA.nbS):
        initial_prob = round(RA.initial[i],2)
        final_prob = round(RA.final[i],2)
        dot.node(str(i),'{} : {} : {}'.format(initial_prob,str(i),final_prob))
    for alphabet in RA.transitions.keys():
        for i in range(RA.nbS):
            for j in range(RA.nbS):
                if RA.transitions[alphabet][i,j] > 0:
                    probability = round(RA.transitions[alphabet][i,j],2)
                    dot.edge(str(i), str(j), '{}, {}'.format(alphabet,probability))
    dot.attr('node',shape='none')
    dot.node('')
    dot.edge('','0')
    dot.render()

def emphasizePNG(RA, string):
    current_node = 0
    str_idx = 0
    dot = Digraph(comment='PFA', format='png')
    for i in range(RA.nbS):
        initial_prob = round(RA.initial[i],2)
        final_prob = round(RA.final[i],2)
        if i == current_node:
            if str_idx < len(string):
                current_node = np.nonzero(RA.transitions[string[str_idx]][current_node])[0]
                str_idx += 1
            dot.node(str(i),'{} : {} : {}'.format(initial_prob,str(i),final_prob),style='filled',fillcolor='yellow')
        else:
            dot.node(str(i),'{} : {} : {}'.format(initial_prob,str(i),final_prob))
    
    current_node = 0
    str_idx = 0
    for alphabet in RA.transitions.keys():
        for i in range(RA.nbS):
            for j in range(RA.nbS):
                if RA.transitions[alphabet][i,j] > 0:
                    if i == current_node and str_idx < len(string):
                        current_node = j
                        str_idx += 1
                        probability = round(RA.transitions[alphabet][i,j],2)
                        dot.edge(str(i), str(j), '{}, {}'.format(alphabet,probability),style='bold')
                    else:   
                        probability = round(RA.transitions[alphabet][i,j],2)
                        dot.edge(str(i), str(j), '{}, {}'.format(alphabet,probability),style='dotted')
    dot.attr('node',shape='none')
    dot.node('')
    dot.edge('','0',style='bold')
    dot.render() 

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mywindow = MyWindow()
    mywindow.show()
    app.exec_()